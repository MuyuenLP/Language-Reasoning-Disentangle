import os
import glob
import fire
import time
from typing import Union

import pandas as pd
import torch
from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager, ReasoningParser

from mlrs.vllms.ForSteering_vllm import ForSteeringVLLM
from mlrs.vllms.ForSteering_vllm_Lens import ForSteeringVLLMLens

from mlrs.lib._df import read_file, save_file
from freeEvalLM.src.Task import TaskLoader

from vllm import ModelRegistry
from mlrs.vllms.steer_qwen2_vllm_lens import LensSteerQwen2ForCausalLM


def main(
    model_name_or_path: str,
    vector_path: str,
    steering_strength: Union[float, list],
    output_dir: str,
    task: str = None,
    file_path: str = None,
    sample: int = -1,
    prompt_key: str = "prompt",
    response_key: str = "response",
    reasoning_key: str = "reasoning",
    reasoning_parser: str = None,
    temperature: float = 0,
    top_p: float = 1.0,
    n_votes: int = 1,
    max_tokens: int = 16384,
    max_model_lens: int = 16384,
    if_continue_generate_for_LLMs: bool = False,
    if_continue_generate_for_LRMs: bool = False,
    tensor_parallel_size: int = 1,
    if_steer_activations: bool = True,
    steering_level: str = "prompt",
    steering_method: str = "vanilla",
    reasoning: bool = True,
    steering_layers: list = None,
    steering_layers2: list = None,
    sys_path: str = None,
    
):
    total_start_time = time.time()
    
    print(n_votes)
    print(tensor_parallel_size)
    print("steering_layers", steering_layers)
    print("Loading model...")
    
    # 记录模型加载开始时间
    model_load_start_time = time.time()
    
    if steering_method == "Lens" and steering_level=="all":
        ModelRegistry.register_model("Qwen2ForCausalLM", LensSteerQwen2ForCausalLM)
        fsvllm = ForSteeringVLLM(model_name_or_path=model_name_or_path, temperature=temperature, top_p=top_p, tensor_parallel_size=tensor_parallel_size, max_tokens=max_tokens, max_model_lens=max_model_lens, steering_layers=steering_layers, steering_level=steering_level)
    else:
        raise ValueError(f"Unknown steering method: {steering_method}")

    model_load_end_time = time.time()
    model_load_time = model_load_end_time - model_load_start_time

    if file_path is None:
        print(f"Using predefined tasks: [{task}]")
        taskLoader = TaskLoader(task, output_dir, sample)
        all_dfs = taskLoader.load_dataset()
        evaluator = taskLoader.load_evaluator()
    else:
        try:
            taskLoader = TaskLoader(task, output_dir, sample)
            evaluator = taskLoader.load_evaluator()
        except:
            pass

        if os.path.isdir(file_path):
            subtasks_data_path = glob.glob(os.path.join(file_path, "*.json"))
            all_dfs = []
            for data_path in subtasks_data_path:
                subtask_name = os.path.basename(data_path).split(".")[0]
                subtask_data = read_file(data_path)
    
                if sample > 0:
                    subtask_data = subtask_data.head(sample).reset_index(drop=True)

                _dict = {
                    "subtask_name": subtask_name,
                    "subtask_data": subtask_data
                }

                all_dfs.append(_dict)

        elif os.path.isfile(file_path):
            if sample > 0:
                df = read_file(file_path)
            else:
                df = read_file(file_path).head(sample)

            all_dfs = [{
                "subtask_name": os.path.basename(file_path).split(".")[0],
                "subtask_data": df
            }]
            
    
    all_querys = []
    big_df = pd.DataFrame()
    for item in all_dfs:

        df = item["subtask_data"]
        subtask_name = item["subtask_name"]
        
        if n_votes > 1:
            df = pd.concat([df] * n_votes, ignore_index=True)
            print(df)
                    
                    
        print(f"running on {subtask_name}")

        query_lines = df[prompt_key].values.tolist()
        length = len(query_lines)
        all_querys += query_lines

        subtask_names = [subtask_name] * length
        subtask_names_df = pd.DataFrame(subtask_names, columns=["subtask_name"])
        
        if hasattr(df, 'subtask_name'):
            pass
        else:
            df = df.join(subtask_names_df)


        big_df = pd.concat([big_df, df], ignore_index=True)

    df = big_df
    query_lines = all_querys
    
    
    if reasoning and sys_path is None:    


        '''
        reasoning_parser_obj = ReasoningParserManager.get_reasoning_parser(reasoning_parser)(fsvllm.tokenizer)
        think_start_token = reasoning_parser_obj.think_start_token
        think_end_token = reasoning_parser_obj.think_end_token
        solution_start_token = reasoning_parser_obj.solution_start_token if hasattr(reasoning_parser_obj, "solution_start_token") else ""

        '''

        think_start_token = "<think>"
        think_end_token = "</think>"

        msg_list = [[{'role': 'user', 'content': d}] for  d in  query_lines]
        # if "Qwen3" in model_name_or_path:
        #     prompt_list = fsvllm.tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
        #     for i in range(len(prompt_list)):
        #         prompt_list[i] = prompt_list[i] + "<think>\n\n"
        # else:
        #     prompt_list = fsvllm.tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
        prompt_list = fsvllm.tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
        prompt_token_ids_list = [fsvllm.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_list]
    
    else:
        with open(sys_path, "r") as f:
            sys_prompt = f.read()
        msg_list = [[{'role': 'system', 'content': f"{sys_prompt}"}, {'role': 'user', 'content': d}] for  d in  query_lines] 
        prompt_token_ids_list = fsvllm.tokenizer.apply_chat_template(msg_list, add_generation_prompt=True)


    fsvllm.init_vector(vector_path, steering_strength)

    inference_start_time = time.time()

    resps = fsvllm.generate_token(if_return_activations=False, if_steer_activations=if_steer_activations, prompt_token_ids_list=prompt_token_ids_list) 
    
    if reasoning:
        reasonings = []
        responses = []
        for i in range(len(resps)):
            resp = resps[i]
            split_res = resp.split(think_end_token)
            if len(split_res) > 1:
                reasoning = split_res[0]
                response = split_res[-1]
            else:
                reasoning = resp
                response = resp  
            
            reasonings.append(reasoning)
            responses.append(response)
    else:
        reasonings = resps
        responses = resps

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    resn_df = pd.DataFrame(reasonings, columns=[reasoning_key])
    resp_df = pd.DataFrame(responses, columns=[response_key])
    big_df = big_df.drop([reasoning_key] + [response_key], axis=1, errors='ignore')
    big_df = big_df.join(resn_df)
    big_df = big_df.join(resp_df)
    df = big_df

    for item in all_dfs:
        subtask_name = item["subtask_name"]
        sub_df = df[df['subtask_name'] == subtask_name]

        result_path = os.path.join(output_dir, subtask_name + ".json")
        os.makedirs(os.path.split(result_path)[0], exist_ok=True)
        save_file(sub_df, result_path)

    evaluator.load_results()
    evaluator.evaluate()
    
if __name__ == "__main__":
    fire.Fire(main)