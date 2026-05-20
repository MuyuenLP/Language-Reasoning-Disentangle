import os
import re
import contextlib
import functools
from typing import List, Tuple, Callable, FrozenSet

# mlrs reaches into the V0 executor graph: `llm_engine.model_executor.driver_worker`.
# vLLM 0.9+ defaults to V1 (`VLLM_USE_V1=1`); with V1 multiprocessing the engine
# omits `model_executor`, which breaks steering hooks. Default to V0 unless the
# user explicitly chose V1.
if "VLLM_USE_V1" not in os.environ:
    os.environ["VLLM_USE_V1"] = "0"

import ray
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# from mlrs.vllms.thu_vllm import create_llm

class ForSteeringVLLM:
    @staticmethod
    def _fwd_input_last_token_positions(all_indices):
        """Match ``def_hook_fn`` indexing against vLLM forward args (flattened positions)."""
        zero_indices = (all_indices == 0).nonzero(as_tuple=True)[0]
        zero_indices = zero_indices - 1
        zero_indices = zero_indices.tolist()
        return zero_indices[1:] + [all_indices.shape[0] - 1]

    def __init__(
            self, 
            model_name_or_path: str,
            temperature: float,
            top_p: float,
            max_model_lens: int = 16384,
            max_tokens: int = 16384,
            tensor_parallel_size: int = 1,
            steering_layers: list = None,
            steering_layers2: list = None,
            steering_level: str = "prompt"
        ):            
        if steering_level == "prompt":
            self.model = LLM(
                model=model_name_or_path,
                dtype="bfloat16",
                max_model_len=max_model_lens,
                tensor_parallel_size=tensor_parallel_size,
                # enforce_eager=True,
            )
            self.lm_model = self.model.llm_engine.model_executor.driver_worker.model_runner.model
            self.model.llm_engine.model_executor.driver_worker.model_runner.return_hidden_states = True
        # elif steering_level == "all":
        #     self.model = None
    
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=1)
        self.steering_layers = steering_layers
        self.steering_layers2 = steering_layers2
        self.steering_level = steering_level
        self.max_model_lens = max_model_lens
        self.tensor_parallel_size = tensor_parallel_size
        self.batch_size = "auto"
        self.all_res_activations = []
        self.layer_num = 0
        self.fn_num = 0
        self.test_num = 0
        self._capture_model_fwd_handles: List = []
        if steering_level == "prompt" :
            self.init_hooks_return_activations()
    
    def init_hooks_return_activations(self):
        self.hook_return_activations = []
        for layer in self.lm_model.model.layers:
            self.hook_return_activations.append(
                (layer, self.def_hook_fn)
            )
            self.layer_num += 1

    
    def reset_hooks_return_activations(self):
        """
        Reset the hooks for collecting activations.
        """
        self.hook_return_activations = []
        self.layer_num = 0
        self.fn_num = 0
        self.test_num = 0
        self.all_res_activations = []
        
        
    def init_hooks_steer_activations(self):
        self.hook_steer_activations = []
        self.layer_num = 0
        for layer_idx, layer in enumerate(self.lm_model.model.layers):
            steer_hook = functools.partial(self.steer_layer_forward_hook, layer_idx)
            self.hook_steer_activations.append((layer, steer_hook))
            self.layer_num += 1

        if self.steering_layers is None:
            self.steering_layers = [i for i in range(len(self.lm_model.model.layers))]
                
    def init_vector(self, vector_path: str, steering_strength: float):
        """
        Initialize the vector for steering.
        """
        self.vector = torch.load(vector_path)
        self.steering_strength = steering_strength
        
        
        # if self.steering_level == "all":
        #     self.model = create_llm(
        #         steering_strength=steering_strength,
        #         steering_vector_path=vector_path,
        #         model_path=self.model_name_or_path,
        #         max_model_lens=self.max_model_lens,
        #         tensor_parallel_size=self.tensor_parallel_size
        #     )

                
    @staticmethod
    def is_end_of_decoder_layer(s):
        pattern = r'^model\.layers\.\d+\.post_attention_layernorm$'
        return bool(re.match(pattern, s))
    

    def steer_layer_forward_hook(self, layer_idx: int, module, input, output):
        out = self._steer_modify_layer(layer_idx, module, input, output)
        self._maybe_capture_steered(layer_idx, out[0], input)
        return out

    def _steer_modify_layer(self, layer_idx: int, module, input, output):
        if self.steering_strength == 0:
            return output

        if layer_idx not in self.steering_layers:
            return output

        res = output[0].clone()
        self.vector = self.vector.to(res.device)
        new_tensor = res + self.steering_strength * self.vector[layer_idx]
        return (new_tensor,) + output[1:]

    def _steered_layer_indices_ordered(self) -> List[int]:
        return sorted(set(self.steering_layers))

    def _capture_steered_layer_set_for_run(self) -> FrozenSet[int]:
        return frozenset(self._steered_layer_indices_ordered())

    def _row_indices_for_capture(
        self, all_indices: torch.Tensor, hidden_tensor: torch.Tensor
    ) -> list[int]:
        """Row indices into ``hidden_tensor``; fall back to 0..N-1 when vLLM omits seq markers."""
        lp = self._fwd_input_last_token_positions(all_indices)
        hr = int(hidden_tensor.shape[0])
        if self._steered_capture_prefill_mode():
            return lp if lp else []
        if len(lp) == hr:
            return lp
        return list(range(hr))

    def _steered_capture_capacity_reached(self) -> bool:
        cap = getattr(self, "_steered_capture_num_prompts", None)
        if cap is None or cap <= 0:
            return False
        return getattr(self, "_steered_capture_rows_collected", 0) >= cap

    def _steered_capture_prefill_mode(self) -> bool:
        """
        At steering strength 0, vLLM often never emits small decode slabs; capture the
        last prefill token per sequence instead (``MLRS_STEER_CAPTURE_MODE=prefill_last``).
        """
        mode = os.environ.get("MLRS_STEER_CAPTURE_MODE", "").strip().lower()
        if mode == "prefill_last":
            return True
        if mode == "decode_first":
            return False
        s1 = float(getattr(self, "steering_strength", 0) or 0)
        s2 = float(getattr(self, "steering_strength2", s1) or 0)
        return s1 == 0.0 and s2 == 0.0

    def _steered_capture_mode_label(self) -> str:
        return "prefill_last" if self._steered_capture_prefill_mode() else "decode_first"

    def _should_capture_forward(
        self, fwd_input_tuple: tuple, hidden_tensor: torch.Tensor
    ) -> bool:
        """
        Decide whether this layer-0 forward should contribute steered activations.

        - Layout match: one hidden row per detected sequence (classic decode).
        - vLLM compact slab: flattened length == hidden rows, much smaller than peak
          prefill chunk (sequence markers in ``positions`` are often missing).

        Collection stops after ``steered_capture_num_prompts`` rows (first decode token
        per prompt across vLLM compact slabs); later decode steps are skipped.
        """
        if getattr(self, "_steered_capture_complete", False):
            return False
        if self._steered_capture_capacity_reached():
            return False
        if not fwd_input_tuple:
            return False
        all_indices = fwd_input_tuple[0]
        if not isinstance(all_indices, torch.Tensor):
            return False
        if all_indices.ndim != 1 or all_indices.numel() == 0:
            return False

        pos_len = int(all_indices.shape[0])
        hr = int(hidden_tensor.shape[0])
        if hr != pos_len:
            return False

        n_seq = len(self._fwd_input_last_token_positions(all_indices))

        if self._steered_capture_prefill_mode():
            return n_seq > 0 and pos_len > n_seq

        if n_seq == hr:
            return True

        peak = getattr(self, "_steered_peak_pos_len", 0)
        if pos_len > peak:
            self._steered_peak_pos_len = pos_len
        peak = self._steered_peak_pos_len

        ratio = float(os.environ.get("MLRS_STEER_CAPTURE_COMPACT_RATIO", "0.08"))
        compact_max = int(os.environ.get("MLRS_STEER_CAPTURE_COMPACT_MAX", "512"))
        min_peak = int(os.environ.get("MLRS_STEER_CAPTURE_MIN_PEAK", "1000"))
        start_threshold = min(compact_max, max(int(peak * ratio), 1))
        continue_ratio = float(
            os.environ.get("MLRS_STEER_CAPTURE_CONTINUE_RATIO", "0.2")
        )

        if getattr(self, "_steered_capture_started", False):
            # Same decode generation step often uses several slabs (e.g. 538 then ~6k
            # tokens); only the first slab is below continue_ratio — keep capturing any
            # post-prefill slab until num_prompts is filled.
            return peak >= min_peak and pos_len < peak

        if peak >= min_peak and pos_len <= start_threshold:
            self._steered_capture_started = True
            return True

        if peak >= min_peak and pos_len <= max(int(peak * continue_ratio), start_threshold):
            self._steered_capture_started = True
            return True

        return False

    def _track_model_forward_for_capture(
        self, layer_idx: int, fwd_input_tuple: tuple, hidden_tensor: torch.Tensor
    ) -> None:
        """
        Count model forwards via decoder layer-0 hooks and gate capture per forward.

        vLLM often never invokes ``model.register_forward_pre_hook``; layer hooks always
        run (steering depends on them). Multiple compact decode slabs are accumulated.
        """
        if not getattr(self, "_steered_capture_cli_requested", False):
            return
        if layer_idx != 0:
            return
        if getattr(self, "_steered_capture_complete", False):
            self._steered_capture_allow_fwd = False
            return

        env_idx = os.environ.get("MLRS_STEER_CAPTURE_MODEL_FWD_INDEX", "").strip()
        if env_idx.isdigit():
            target_idx = int(env_idx)
            self._steered_capture_allow_fwd = self._model_fwd_seen == target_idx
        else:
            self._steered_capture_allow_fwd = self._should_capture_forward(
                fwd_input_tuple, hidden_tensor
            )

        if len(self._steered_capture_fwd_debug) < 24:
            all_indices = fwd_input_tuple[0] if fwd_input_tuple else None
            pos_len = (
                int(all_indices.shape[0])
                if isinstance(all_indices, torch.Tensor)
                else None
            )
            n_seq = (
                len(self._fwd_input_last_token_positions(all_indices))
                if isinstance(all_indices, torch.Tensor)
                else None
            )
            self._steered_capture_fwd_debug.append(
                {
                    "fwd": self._model_fwd_seen,
                    "allow": self._steered_capture_allow_fwd,
                    "pos_len": pos_len,
                    "n_seq": n_seq,
                    "hidden_rows": int(hidden_tensor.shape[0]),
                    "peak_pos_len": getattr(self, "_steered_peak_pos_len", 0),
                }
            )

        self._model_fwd_seen += 1

    def _maybe_capture_steered(
        self,
        layer_idx: int,
        hidden_tensor,
        fwd_input_tuple: tuple,
    ):
        if not getattr(self, "_steered_capture_cli_requested", False):
            return
        self._track_model_forward_for_capture(layer_idx, fwd_input_tuple, hidden_tensor)
        if not getattr(self, "_steered_capture_allow_fwd", False):
            return
        if getattr(self, "_steered_capture_complete", False):
            return
        layers_set = getattr(self, "capture_steered_layer_set", frozenset())
        if layer_idx not in layers_set:
            return
        all_indices = fwd_input_tuple[0]
        row_indices = self._row_indices_for_capture(all_indices, hidden_tensor)
        n_layers = getattr(self, "_steered_capture_num_layers", 1) or 1
        self.capture_steered_hook_tick += 1

        res = hidden_tensor
        row_offset = self.prompt_index_steered_capture
        n_batch = len(row_indices)
        cap = getattr(self, "_steered_capture_num_prompts", None)
        if cap is not None and cap > 0:
            remaining = cap - row_offset
            if remaining <= 0:
                self._steered_capture_complete = True
                return
            n_batch = min(n_batch, remaining)
            row_indices = row_indices[:n_batch]

        for i in range(n_batch):
            if self.capture_steered_hook_tick % n_layers == 1:
                self.temp_steered_activations.append([])
            a = row_indices[i]
            self.temp_steered_activations[i + row_offset].append(res[a].detach().cpu())

        if self.capture_steered_hook_tick % n_layers == 0:
            self.prompt_index_steered_capture += n_batch
            self._steered_capture_rows_collected = (
                getattr(self, "_steered_capture_rows_collected", 0) + n_batch
            )
            self._steered_capture_allow_fwd = False
            if self._steered_capture_capacity_reached():
                self._steered_capture_complete = True

    def _finalize_steered_activations(self):
        stacked = getattr(self, "temp_steered_activations", None)
        if not stacked:
            return None
        n_layers = getattr(self, "_steered_capture_num_layers", None)
        out_rows = []
        for i in range(len(stacked)):
            if len(stacked[i]) == 0:
                continue
            if n_layers is not None and len(stacked[i]) != n_layers:
                continue
            out_rows.append(torch.stack(stacked[i]))
        if not out_rows:
            return None
        cap = getattr(self, "_steered_capture_num_prompts", None)
        if cap is not None and cap > 0 and len(out_rows) > cap:
            out_rows = out_rows[:cap]
        return torch.stack(out_rows)

    def return_steered_activations_tensor(self):
        """
        Dense tensor ``[num_prompts, num_steered_layers, hidden_dim]`` from the last
        ``generate_token`` run with ``capture_steered_first_token=True``.

        Indices follow ``capture_steered_layer_indices`` (sorted intervened decoder ids).
        """
        return getattr(self, "_steered_activation_tensor", None)

    def steered_activation_layer_indices(self):
        """Ordered decoder indices aligned with activation dim ``1``."""
        ix = getattr(self, "capture_steered_layer_indices_ordered", None)
        return [] if ix is None else list(ix)
    
    
    def def_hook_fn(self, moudle, input, output):
        '''
        output:
        (
            hidden_states,      # shape:
            maybe_kv_cache      # shape: depends on implementation, often [batch_size, num_heads, seq_len, head_dim]
        )

        '''
        res = output[0]
        all_indices = input[0]
        last_token_positions = self._fwd_input_last_token_positions(all_indices)

        prompt_index_before = self.prompt_index
 
        
        self.fn_num += 1 
        # for self.prompt_index in range(self.prompt_index, self.prompt_index + len(last_token_positions)):
        for i in range(len(last_token_positions)):
            if self.fn_num % self.layer_num == 1:
                self.temp_activations.append([])
                self.test_num += 1
            else:
                pass
            a = last_token_positions[i] 
            self.temp_activations[i + self.prompt_index].append(res[a].detach().cpu())
        

        if self.fn_num % self.layer_num == 0:
            self.prompt_index += len(last_token_positions)
        


        
        
    def return_activations(self):
        """
        Return the activations collected during the forward pass.
        """
        return torch.stack( self.all_res_activations)

    @contextlib.contextmanager
    def add_hooks(
        self,
        module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
        module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
        **kwargs
    ):
        """
        Context manager for temporarily adding forward hooks to a model.

        Parameters
        ----------
        module_forward_pre_hooks
            A list of pairs: (module, fnc) The function will be registered as a
                forward pre hook on the module
        module_forward_hooks
            A list of pairs: (module, fnc) The function will be registered as a
                forward hook on the module
        """
        try:
            handles = []
            for module, hook in module_forward_pre_hooks:
                partial_hook = functools.partial(hook, **kwargs)
                handles.append(module.register_forward_pre_hook(partial_hook))
            for module, hook in module_forward_hooks:
                partial_hook = functools.partial(hook, **kwargs)
                handles.append(module.register_forward_hook(partial_hook))
            yield
        finally:
            for h in handles:
                h.remove()
    

    def generate(
        self, 
        prompts : List[str] = None 
    ):
        outputs = self.model.generate(
            prompts,
            sampling_params=self.sampling_params,
            use_tqdm=True
        )
        return outputs

    def generate_token(
        self, 
        if_return_activations: bool = True,
        if_steer_activations: bool = False,
        capture_steered_first_token: bool = False,
        prompt_token_ids_list : List[str] = None,
        steered_capture_num_prompts: int | None = None,
    ):
        self._steered_activation_tensor = None
        self.capture_steered_layer_indices_ordered = None

        if if_return_activations:
            self.temp_activations = []
            self.prompt_index = 0
            with self.add_hooks([], self.hook_return_activations):
                outputs = self.model.generate(
                    prompt_token_ids=prompt_token_ids_list,
                    sampling_params=self.sampling_params,
                    use_tqdm=True if self.batch_size == "auto" else False
                )

            for i in range(len(self.temp_activations)):
                if len(self.temp_activations[i]) == 0:
                    continue
                self.temp_activations[i] = torch.stack(self.temp_activations[i])

            self.all_res_activations.extend(self.temp_activations)
            
        elif if_steer_activations and self.steering_level == "prompt":
            capture_run = bool(capture_steered_first_token)
            try:
                if capture_steered_first_token:
                    self._prepare_steered_capture_state(
                        num_prompts=steered_capture_num_prompts
                    )
                with self.add_hooks([], self.hook_steer_activations):
                    outputs = self.model.generate(
                        prompt_token_ids=prompt_token_ids_list,
                        sampling_params=self.sampling_params,
                        use_tqdm=True if self.batch_size == "auto" else False
                    )
            finally:
                if capture_run:
                    self._finalize_steered_capture_after_generate()

        else:
            outputs = self.model.generate(
                prompt_token_ids=prompt_token_ids_list,
                sampling_params=self.sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
            )

        answers = []
        for output in outputs:
            answers.append(output.outputs[0].text)
        return answers

    def _prepare_steered_capture_state(self, num_prompts: int | None = None):
        ordered = list(self._steered_layer_indices_ordered())
        env_idx = os.environ.get("MLRS_STEER_CAPTURE_MODEL_FWD_INDEX", "").strip()
        target_fwd = 1
        if env_idx.isdigit():
            target_fwd = int(env_idx)

        self._steered_capture_cli_requested = True
        self.capture_steered_layer_indices_ordered = ordered
        self.capture_steered_layer_set = self._capture_steered_layer_set_for_run()
        self._steered_capture_num_layers = max(len(ordered), 1)
        self._steered_capture_num_prompts = num_prompts
        self.temp_steered_activations = []
        self.prompt_index_steered_capture = 0
        self.capture_steered_hook_tick = 0
        self._model_fwd_seen = 0
        self.steered_capture_model_fwd_index = target_fwd
        self._steered_capture_allow_fwd = False
        self._steered_peak_pos_len = 0
        self._steered_capture_started = False
        self._steered_capture_rows_collected = 0
        self._steered_capture_complete = False
        self._steered_capture_fwd_debug: list[dict[str, object]] = []

    def _finalize_steered_capture_after_generate(self):
        self._steered_capture_cli_requested = False
        if getattr(self, "capture_steered_layer_indices_ordered", None) is None:
            self._steered_activation_tensor = None
            return
        stacked = self._finalize_steered_activations()
        self._steered_activation_tensor = stacked
        if stacked is None:
            import sys

            env_idx = os.environ.get("MLRS_STEER_CAPTURE_MODEL_FWD_INDEX", "").strip()
            dbg = getattr(self, "_steered_capture_fwd_debug", None)
            dbg_s = f" sample_forwards={dbg}" if dbg else ""
            print(
                "[ForSteeringVLLM] steer capture empty after generate: "
                f"model_forwards_seen={getattr(self, '_model_fwd_seen', 0)}, "
                f"rows_collected={getattr(self, '_steered_capture_rows_collected', 0)}, "
                f"peak_pos_len={getattr(self, '_steered_peak_pos_len', 0)}, "
                f"MLRS_STEER_CAPTURE_MODEL_FWD_INDEX={env_idx!r} "
                f"(unset = compact-slab auto detect; try FWD_INDEX=6 or tune "
                "MLRS_STEER_CAPTURE_COMPACT_MAX / COMPACT_RATIO)."
                f"{dbg_s}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    pass