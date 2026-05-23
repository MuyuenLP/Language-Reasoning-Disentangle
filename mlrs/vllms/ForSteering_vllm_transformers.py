"""Variant of [`ForSteeringVLLM`][mlrs.vllms.ForSteering_vllm.ForSteeringVLLM] with custom hook/generate behavior."""

from typing import List

import torch

from mlrs.vllms.ForSteering_vllm import ForSteeringVLLM


class ForSteeringVLLMTransformers(ForSteeringVLLM):
    def def_hook_fn(self, moudle, input, output):
        self.fn_num += 1
        if self.fn_num % self.layer_num == 1:
            self.temp_activations.append([])
            self.test_num += 1
        res = output[0][0][-1]
        self.temp_activations[0].append(res.detach().cpu())

    def generate_token(
        self,
        if_return_activations: bool = True,
        if_steer_activations: bool = False,
        prompt_token_ids_list: List[str] = None,
    ):
        if if_return_activations:
            self.temp_activations = []
            self.prompt_index = 0
            with self.add_hooks([], self.hook_return_activations):
                outputs = self.model.generate(
                    prompt_token_ids=prompt_token_ids_list,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

            for i in range(len(self.temp_activations)):
                if len(self.temp_activations[i]) == 0:
                    continue
                self.temp_activations[i] = torch.stack(self.temp_activations[i])

            self.all_res_activations.extend(self.temp_activations)

        elif if_steer_activations:
            with self.add_hooks([], self.hook_steer_activations):
                outputs = self.model.generate(
                    prompt_token_ids=prompt_token_ids_list,
                    sampling_params=self.sampling_params,
                    use_tqdm=True if self.batch_size == "auto" else False,
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
