import os
import re
import contextlib
import functools
from typing import List, Tuple, Callable

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


from mlrs.vllms.ForSteering_vllm import ForSteeringVLLM

class ForSteeringVLLMLens(ForSteeringVLLM):
    
    def projection(self, emb, lang_dir):
        
        lang_dir_norm = lang_dir / torch.linalg.norm(lang_dir, axis=1, keepdims=True).to(emb.dtype)
        proj = torch.matmul(emb, lang_dir_norm.T)
        
        return torch.matmul(proj, lang_dir_norm)
        

    def init_vector(self, vector_path: str, steering_strength: list):
        """
        Initialize the vector for steering.
        """

        self.vector = torch.load(vector_path)
        self.steering_strength = steering_strength[0]
        self.steering_strength2 = steering_strength[1]
        
                    

    def def_hook_steer_fn(self, moudle, input, output):
        '''
        output:
        (
            hidden_states,      # shape: [batch_size, seq_len, hidden_dim]
            maybe_kv_cache      # shape: depends on implementation, often [batch_size, num_heads, seq_len, head_dim]
        )

        '''
        res = output[0].clone()

        self.fn_num += 1 
        
        

        self.fn_num += 1 
        layer = self.fn_num % self.layer_num - 1
        if layer < 0:
            layer = self.layer_num - 1
            
        if layer in self.steering_layers:
            if self.steering_strength == 0:
                return output
            self.layer_space = self.vector.to(res.device)[layer]
            proj = self.projection(res, self.layer_space.to(torch.bfloat16).to(res.device))
            new_tensor = res - self.steering_strength * proj

            return (new_tensor, ) + output[1:]
        
        elif self.steering_layers2 is not None and layer in self.steering_layers2:
            if self.steering_strength2 == 0:
                return output
            self.layer_space = self.vector.to(res.device)[layer]
            proj = self.projection(res, self.layer_space.to(torch.bfloat16).to(res.device))
            new_tensor = res - self.steering_strength2 * proj

            
            return (new_tensor, ) + output[1:]
        
        else:
            return output
        
        
        