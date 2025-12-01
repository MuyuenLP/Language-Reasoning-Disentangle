import os
import pandas as pd
from freeEvalLM._lib._df import read_file, save_file
from freeEvalLM.src.Evaluator import Evaluator
import re

import re
import json

import torch
import ray

from math_verify import parse, verify
from freeEvalLM.tasks.language_fidelity.fidelity_evaluator import fidelity_evaluator

# from openrlhf.datasets.multiround_prompts_dataset import remove_chat_template

def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"




class mgsm(Evaluator):
    def __init__(self, task, result_dir):
        super().__init__(task, result_dir)
        self.fidelity_evaluator = fidelity_evaluator()

    def extract_answer(self, response_text):
        """
        多语言答案提取方法（支持 bn/en/ja/sw/zh）
        输入示例：
        "答案是 18 美元。" → 输出："18"
        "The answer is 18 dollars." → 输出："18"
        "答えは18です。" → 输出："18"
        """

        # 统一处理空响应
        if not isinstance(response_text, str):
            return "[invalid]"


        # 匹配 LaTeX \boxed{...} 格式
        # boxed_pattern = r'\\boxed\{(.*?)\}'
        # match = re.search(boxed_pattern, response_text)
        # print("#################")
        # print(match)
        # if match:
        #     answer = match.group(1).strip()
        #     return answer
    
        find = extract_boxed_content(response_text)
        answer = find
        if answer != "None":
            # print("##############")
            # print(answer)
            return answer

    
        pattern = r'答案是 (\-?[0-9\.\,]+)| \
            The answer is (\-?[0-9\.\,]+)| \
            答えは (\-?[0-9\.\,]+)| \
            Jibu ni (\-?[0-9\.\,]+)| \
            হল (\-?[0-9\.\,]+)| \
            Die Antwort lautet (\-?[0-9\.\,]+)| \
            La respuesta es (\-?[0-9\.\,]+)| \
            La réponse est (\-?[0-9\.\,]+)| \
            Ответ — (\-?[0-9\.\,]+)| \
            คำตอบคือ (\-?[0-9\.\,]+)'


        matches = re.findall(pattern, response_text, re.IGNORECASE | re.UNICODE)
        # print(matches)

        if matches:
            # 选择最后一个非空匹配项
            for match in reversed(matches):  # 从后往前遍历
                for group in reversed(match):  # 确保从最后一个非空匹配项开始取值
                    if group:
                        return group

        '''
        添加后处理 在捕捉不到任何数字时采用最后一个数
        '''
        fallback_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
        fallback_matches = re.findall(fallback_pattern, response_text)
        print(fallback_matches)

        if fallback_matches:
            # for match in fallback_matches:  # 遍历
            for match in reversed(fallback_matches):  # 从后往前遍历
                for group in reversed(match):
                    if group:
                        return group


        return "[invalid]"

    def filter_answer_subtask(self, data):
        """ 生成符合格式的 filtered_resps """
        filtered = []
        for _, row in data.iterrows():
            # 处理 response 可能是列表或字符串的情况
            if isinstance(row["response"], list):
                response_text = row["response"][0] if row["response"] else ""
            elif isinstance(row["response"], str):
                response_text = row["response"]
            else:
                response_text = ""

            # 提取答案
            answer = self.extract_answer(response_text)

            clean_answer = re.sub(r'\.0+$', '', str(answer).strip())
            clean_answer = re.sub(r'[,\s]', '', clean_answer)  # 移除逗号、空格
            clean_answer = re.sub(r'\.$', '', clean_answer)  # 移除结尾的小数点
            clean_answer = re.sub(r'\$', '', clean_answer)  # 移除数字中的 $

            filtered.append([clean_answer])

        return filtered

    def compare(self, pred_answers, true_answers):
        """ 严格格式匹配 """
        scores = []
        for pred_list, true in zip(pred_answers, true_answers):
            # 提取列表中的第一个答案
            pred = pred_list[0] if isinstance(pred_list, list) and len(pred_list) > 0 else "[invalid]"
            # print("####################3")
            # print(parse(f"${true}$"))
            # print(parse(f"${pred}$"))
            if verify(parse(f"${true}$"), parse(f"${pred}$")):
                score = 1
            else:
                score = 0

            # # 预处理字符串，去掉结尾的小数点和千分位逗号
            # pred = re.sub(r'[,\s]', '', pred)  # 移除逗号、空格
            # pred = re.sub(r'\.$', '', pred)  # 移除结尾的小数点
            # pred = re.sub(r'\$', '', pred)  # 移除数字中的 $
            # 确保数据类型一致
            # try:
            #     pred = float(pred)  # 尝试转换
            # except ValueError:
            #     pred = "[invalid]"  # 转换失败，标记为无效

            scores.append(score)

        return scores
    
    def compute_fidelity(self, inputs, filtered_resps, name):
        return self.fidelity_evaluator.evaluate(inputs, filtered_resps, name)

    def evaluate(self):
        all_names = []
        all_finals = []
        all_rep_fidelity = []
        all_rea_fidelity = []
        print("Evaluating...")
        for subtask, data_path in zip(self.all_dfs, self.subtasks_data_path):
            name = subtask["subtask_name"]
            all_names.append(name)
            data = subtask["subtask_data"]
            print(f"Evaluating {name}")

            filtered_resps = self.filter_answer_subtask(data)
            scores = self.compare(filtered_resps, data["target"])
            rep_fidelities = self.compute_fidelity(data["input"], data["response"], name)
            rea_fidelities =self.compute_fidelity(data["input"], data["reasoning"], name)


            # 计算最终得分
            final_score = sum(scores) / len(scores)
            all_finals.append(final_score)
            final_rep_fidelity = sum(rep_fidelities) / len(rep_fidelities)
            final_rea_fidelity = sum(rea_fidelities) / len(rea_fidelities)
            all_rep_fidelity.append(final_rep_fidelity)
            all_rea_fidelity.append(final_rea_fidelity)

            # 保存处理后的数据
            df = pd.DataFrame({
                'filtered_answer': filtered_resps,
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
    # evaluator = mgsm("mgsm", "/data/works_jhguo/mlrs/results/qwen3-4b/dev_250429/test/generate_strength_0")
    # evaluator.load_results()
    # evaluator.evaluate()
    
    
    
    root_dir = "/data/works_jhguo/mlrs/results/r1-distill-qwen-14b/dev_250429/ultra"
    leaf_dirs = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 如果当前目录下没有子目录，则是最末端文件夹
        if not dirnames:
            leaf_dirs.append(dirpath)

    print(leaf_dirs)
    
    for _dir in leaf_dirs:
        if not os.listdir(_dir):
            continue
        evaluator = mgsm("mgsm", _dir)
        evaluator.load_results()
        evaluator.evaluate()
    
    pass
    
