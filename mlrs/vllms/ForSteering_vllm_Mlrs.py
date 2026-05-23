"""Mlrs-layer steering hooks; inherits from [`ForSteeringVLLM`][mlrs.vllms.ForSteering_vllm.ForSteeringVLLM].

Import [`ForSteeringVLLMMlrs`][mlrs.vllms.ForSteering_vllm_Mlrs.ForSteeringVLLMMlrs]
only after/for ``ForSteering_vllm`` so ``VLLM_USE_V1`` defaults are applied before ``vllm`` loads.
"""

import torch

from mlrs.vllms.ForSteering_vllm import ForSteeringVLLM


class ForSteeringVLLMMlrs(ForSteeringVLLM):
    def projection(self, emb, lang_dir):
        lang_dir_norm = lang_dir / torch.linalg.norm(lang_dir, axis=1, keepdims=True).to(
            emb.dtype
        )
        proj = torch.matmul(emb, lang_dir_norm.T)

        return torch.matmul(proj, lang_dir_norm)

    def init_vector(self, vector_path: str, steering_strength):  # float | Sequence[float]
        """
        Initialize the steering vector.

        Accepts either a single scalar (applied to ``steering_layers`` and
        ``steering_layers2``) or a sequence ``(strength_mid, strength_late)``
        for the two Mlrs layer groups.
        """
        self.vector = torch.load(vector_path)
        if isinstance(steering_strength, (list, tuple)):
            seq = [float(x) for x in steering_strength]
            if len(seq) == 0:
                raise ValueError("steering_strength sequence must not be empty")
            if len(seq) == 1:
                self.steering_strength = self.steering_strength2 = seq[0]
            else:
                self.steering_strength = seq[0]
                self.steering_strength2 = seq[1]
        else:
            s = float(steering_strength)
            self.steering_strength = self.steering_strength2 = s

    def _steered_layer_indices_ordered(self):
        u = set(self.steering_layers)
        if getattr(self, "steering_layers2", None):
            u |= set(self.steering_layers2)
        return sorted(u)

    def _steer_modify_layer(self, layer_idx, module, input, output):
        if layer_idx in self.steering_layers:
            if self.steering_strength == 0:
                return output
            res = output[0].clone()
            self.vector = self.vector.to(res.device)
            layer_space = self.vector[layer_idx]
            proj = self.projection(
                res, layer_space.to(torch.bfloat16).to(res.device)
            )
            new_tensor = res - self.steering_strength * proj
            return (new_tensor,) + output[1:]

        if self.steering_layers2 is not None and layer_idx in self.steering_layers2:
            if self.steering_strength2 == 0:
                return output
            res = output[0].clone()
            self.vector = self.vector.to(res.device)
            layer_space = self.vector[layer_idx]
            proj = self.projection(
                res, layer_space.to(torch.bfloat16).to(res.device)
            )
            new_tensor = res - self.steering_strength2 * proj
            return (new_tensor,) + output[1:]

        return output

