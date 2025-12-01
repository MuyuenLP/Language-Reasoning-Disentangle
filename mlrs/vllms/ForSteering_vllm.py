import os
import re
import contextlib
import functools
from typing import List, Tuple, Callable

import ray
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from mlrs.vllms.thu_vllm import create_llm

class ForSteeringVLLM:
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
        elif steering_level == "all":
            self.model = None
    
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
        for layer in self.lm_model.model.layers:
            self.hook_steer_activations.append(
                (layer, self.def_hook_steer_fn)
            )
            self.layer_num += 1

        if self.steering_layers is None:
            self.steering_layers = [i for i in range(len(self.lm_model.model.layers))]
                
    def init_vector(self, vector_path: str, steering_strength: float):
        """
        Initialize the vector for steering.
        """
        self.vector = torch.load(vector_path)
        self.steering_strength = steering_strength
        
        
        if self.steering_level == "all":
            self.model = create_llm(
                steering_strength=steering_strength,
                steering_vector_path=vector_path,
                model_path=self.model_name_or_path,
                max_model_lens=self.max_model_lens,
                tensor_parallel_size=self.tensor_parallel_size
            )

                
    @staticmethod
    def is_end_of_decoder_layer(s):
        pattern = r'^model\.layers\.\d+\.post_attention_layernorm$'
        return bool(re.match(pattern, s))
    

    def def_hook_steer_fn(self, moudle, input, output):
        '''
        output:
        (
            hidden_states,      # shape: [batch_size, seq_len, hidden_dim]
            maybe_kv_cache      # shape: depends on implementation, often [batch_size, num_heads, seq_len, head_dim]
        )

        '''
        if self.steering_strength == 0:
            return output
        self.fn_num += 1 
        layer = self.fn_num % self.layer_num - 1
        if layer < 0:
            layer = self.layer_num - 1
            
        if layer in self.steering_layers:
            res = output[0].clone()
            new_shape = res.shape
            self.vector = self.vector.to(res.device)
            new_tensor = res + self.steering_strength * self.vector[self.fn_num % self.layer_num - 1]
            '''
            这里貌似得去掉kv cache
            '''
            # new_output = (new_tensor, output1) + output[1:]
            return (new_tensor, ) + output[1:]
        else:
            return output
    
    
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
        zero_indices = (all_indices == 0).nonzero(as_tuple=True)[0]
        zero_indices = zero_indices - 1
        zero_indices = zero_indices.tolist()
        last_token_positions = zero_indices[1:] + [all_indices.shape[0] - 1]

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
        prompt_token_ids_list : List[str] = None
    ):
        
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
            with self.add_hooks([], self.hook_steer_activations):
                outputs = self.model.generate(
                    prompt_token_ids=prompt_token_ids_list,
                    sampling_params=self.sampling_params,
                    use_tqdm=True if self.batch_size == "auto" else False
                )
                
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


if __name__ == "__main__":
    pass