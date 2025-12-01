import glob, os
from tqdm import tqdm
current_path = os.getcwd()

import re
import pandas as pd

from freeEvalLM._lib._df import read_file, save_file
from freeEvalLM.src.Evaluator import Evaluator

from freeEvalLM.tasks.xwinograd.common import *
from freeEvalLM.tasks.language_fidelity.fidelity_evaluator import fidelity_evaluator

class xwinograd(Evaluator):
    def __init__(self, 
        task, 
        result_dir
        ):
        self.task = task
        self.result_dir = result_dir
        self.none = 0
        self.fidelity_evaluator = fidelity_evaluator()

    def count(self, scores):
        final = round(scores.count(1)/len(scores), 4)
        return final
                    
    
    def compare(self, answers, targets):
        scores = []
        for answer, target in zip(answers, targets):
            if answer == target:
                score = 1
            else:
                score = 0
            scores.append(score)
        return scores

    def compute_fidelity(self, inputs, filtered_resps, name):
        return self.fidelity_evaluator.evaluate(inputs, filtered_resps, name)

    def evaluate(self):
        all_names = []
        all_finals = []
        all_rep_fidelity = []
        all_rea_fidelity = []
        for subtask, data_path in zip(self.all_dfs,  self.subtasks_data_path):
            name = subtask["subtask_name"]
            all_names.append(name)
            data = subtask["subtask_data"]
            
            responses = data["response"] 
            targets = data["target"]
            scores = []
            for response, target in zip(responses, targets):

                response_text = normalize_response(response)
                extracted_answer = None
                for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                    regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                    match = re.search(regex, response_text)
                    if match:
                        extracted_answer = normalize_extracted_answer(match.group(1))
                        break
                score = 1.0 if extracted_answer == target else 0.0
                scores.append(score)

            rep_fidelities = self.compute_fidelity(data["prompt"], data["response"], name)
            rea_fidelities =self.compute_fidelity(data["prompt"], data["reasoning"], name)
            final_rep_fidelity = sum(rep_fidelities) / len(rep_fidelities)
            final_rea_fidelity = sum(rea_fidelities) / len(rea_fidelities)
            all_rep_fidelity.append(final_rep_fidelity)
            all_rea_fidelity.append(final_rea_fidelity)
            
            # answers = self.filter_answer_subtask(data)
            # scores = self.compare(answers, data["target"])
            final = self.count(scores)
            all_finals.append(final)
            df = pd.DataFrame({
                'filtered_answer': responses,
                'score': scores,
                'rep_fidelity': rep_fidelities,
                'rea_fidelity': rea_fidelities
            })
            print(data)
            data = data.join(df)
            save_file(data, os.path.join(os.path.dirname(data_path), f"{name}.json"))
            print(f"Saved {name}.json")

        # 生成最终评估结果
        df = pd.DataFrame({
            'subtask': all_names + ["FINAL"],
            'score': all_finals + [round(sum(all_finals) / len(all_finals), 4)],
            'rep_fidelity': all_rep_fidelity + [round(sum(all_rep_fidelity) / len(all_rep_fidelity), 4)],
            'rea_fidelity': all_rea_fidelity + [round(sum(all_rea_fidelity) / len(all_rea_fidelity), 4)]
        })
        save_file(df, os.path.join(os.path.dirname(data_path), "Results.csv"))
            


if __name__ == "__main__":

    # root_dir = "/data/works_jhguo/mlrs/results/r1-distill-qwen-14b/dev_250429/ultra_xwinograd"
    # leaf_dirs = []

    # for dirpath, dirnames, filenames in os.walk(root_dir):
    #     # 如果当前目录下没有子目录，则是最末端文件夹
    #     if not dirnames:
    #         leaf_dirs.append(dirpath)

    # print(leaf_dirs)
    
    # for _dir in leaf_dirs:
    #     if not os.listdir(_dir):
    #         continue
    #     evaluator = xwinograd("xwinograd", _dir)
    #     evaluator.load_results()
    #     evaluator.evaluate()

    root_dirs = [
        # "/data/works_jhguo/mlrs/results/qwen25-3b-instruct/dev_250507/ultra_xwinograd",
        # "/data/works_jhguo/mlrs/results/qwen25-7b-instruct/dev_250507/ultra_xwinograd",
        # "/data/works_jhguo/mlrs/results/qwen3-1.7b/dev_250505/ultra_xwinograd",
        # "/data/works_jhguo/mlrs/results/qwen3-4b/dev_250505/ultra_xwinograd",
        # "/data/works_jhguo/mlrs/results/qwen3-8b/dev_250505/ultra_xwinograd",
        # "/data/works_jhguo/mlrs/results/r1-distill-qwen-8b/dev_250429/ultra_xwinograd",
        # "/data/works_jhguo/mlrs/results/glm-z1-9b/dev_250429/ultra_xwinograd",
        # "/data/works_jhguo/mlrs/results/qwq/dev_250429/ultra_xwinograd"
        "/data/works_jhguo/mlrs/results/r1-distill-qwen-14b/dev_250429/ultra_xwinograd",
    ]
    
    for root_dir in root_dirs:

        leaf_dirs = []

        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 如果当前目录下没有子目录，则是最末端文件夹
            if not dirnames:
                leaf_dirs.append(dirpath)

        print(leaf_dirs)
        
        for _dir in leaf_dirs:
            if not os.listdir(_dir):
                continue
            evaluator = xwinograd("xwinograd", _dir)
            evaluator.load_results()
            evaluator.evaluate()