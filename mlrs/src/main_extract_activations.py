import os
import glob
import fire
from tqdm import tqdm

import pandas as pd
import torch
from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager, ReasoningParser

from mlrs.vllms.ForSteering_vllm import ForSteeringVLLM
from mlrs.vllms.ForSteering_vllm_transformers import ForSteeringVLLMTransformers
from mlrs.lib._df import read_file, save_file





def main(
    model_name_or_path: str,
    output_dir: str,
    input_path: str,
    if_think: bool = False,
    prompt_key: str = "prompt",
    response_key: str = "response",
    reasoning_key: str = "reasoning",
    reasoning_parser: str = None,
    temperature: float = 0,
    top_p: float = 1.0,
    if_vis: bool = True,
    vis_keys: str = None,
    model_type: str = "vllm",
    if_system: bool = False,
    sys_path: str = "/data/works_jhguo/mlrs/prompts/qwen25.txt",
    if_Lens: bool = False,
    sample: int = -1,
):
    if if_Lens:
        fsvllm = ForSteeringVLLM(model_name_or_path=model_name_or_path, temperature=temperature, top_p=top_p, max_tokens=1)
    else:
        if model_type == "vllm":
            fsvllm = ForSteeringVLLM(model_name_or_path=model_name_or_path, temperature=temperature, top_p=top_p, max_tokens=1)
        elif model_type == "transformers":
            fsvllm = ForSteeringVLLMTransformers(model_name_or_path=model_name_or_path, temperature=temperature, top_p=top_p, max_tokens=1)        

    reasoning_parser_obj = ReasoningParserManager.get_reasoning_parser(reasoning_parser)(fsvllm.tokenizer)
    think_start_token = reasoning_parser_obj.think_start_token
    think_end_token = reasoning_parser_obj.think_end_token
    solution_start_token = reasoning_parser_obj.solution_start_token if hasattr(reasoning_parser_obj, "solution_start_token") else ""

    input_data_df = read_file(input_path)
    
    if sample > 0:
        input_data_df = input_data_df.sample(sample)
        
    prompts = input_data_df[prompt_key].tolist()
    
    if if_think:
        reasonings = input_data_df[reasoning_key].tolist()

    if if_system == False:
        msg_list = [[{'role': 'user', 'content': d}] for  d in  prompts]    
    else:
        with open(sys_path, "r") as f:
            system_prompt = f.read()
        msg_list = [[{'role': 'system', 'content': f"{system_prompt}"}, {'role': 'user', 'content': d}] for  d in  prompts]  
    # if "Qwen3" in model_name_or_path:
    #     prompt_list = fsvllm.tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
    #     for i in range(len(prompt_list)):
    #         prompt_list[i] = prompt_list[i] + "<think>\n\n"
    # else:  
        # prompt_list = fsvllm.tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
    prompt_list = fsvllm.tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
    if if_think:
        prompt_list = [p.rstrip('\n') for p in prompt_list]
        prompt_list = [f'{p}{"" if p.endswith(think_start_token) else think_start_token}\n\n{r}\n\n{think_end_token}{solution_start_token}' for p, r in zip(prompt_list, reasonings)]


    prompt_token_ids_list = [fsvllm.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_list]
    # prompt_token_ids_list = prompt_token_ids_list[0:10]
    
    if model_type == "vllm":
        resp = fsvllm.generate_token(if_return_activations=True, prompt_token_ids_list=prompt_token_ids_list) 
    elif model_type == "transformers":
        resp = []
        for prompt_token_ids in tqdm(prompt_token_ids_list):
            resp.append(fsvllm.generate_token(if_return_activations=True, prompt_token_ids_list=[prompt_token_ids]))    
            
    resp_df = pd.DataFrame(resp, columns=[response_key])
    input_data_df = input_data_df.join(resp_df)
    
    filename = os.path.basename(input_path) 
    filename = os.path.splitext(filename)[0] 
    
    os.makedirs(os.path.join(output_dir, "resp"), exist_ok=True)
    resp_output_path = os.path.join(output_dir, "resp", filename + ".json")
    save_file(input_data_df, resp_output_path)
    
    a = fsvllm.return_activations()
    os.makedirs(os.path.join(output_dir, "acts"), exist_ok=True)
    acts_output_path = os.path.join(output_dir, "acts", filename + ".pt")
    torch.save(a, acts_output_path)
    

    if if_vis:
        if vis_keys is not None:
            pass
    
    

    
    
    
    
if __name__ == "__main__":
    fire.Fire(main)

