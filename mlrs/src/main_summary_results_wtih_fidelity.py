import fire

from transformers import AutoTokenizer

from mlrs.lib._json import *
from mlrs.lib._df import *


import pandas as pd
import openpyxl


def lists_to_excel4(list1, list2, list3, list4, filename='output.xlsx', column_names=("Column1", 'Column2', 'Column3')):
    data = {
        column_names[0]: list1,
        column_names[1]: list2,
        column_names[2]: list3,
        column_names[3]: list4
    }

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)



def get_all_json_files(directory):
    json_files = [os.path.join(directory, f) for f in os.listdir(os.path.join(directory)) if f.endswith('.json')]
    return json_files


def main(
    result_dir: str,
    skip: bool = False
):
    if not skip:
        tokenizer = AutoTokenizer.from_pretrained("~/Downloads/DeepSeek-R1-Distill-Qwen-7B")


        all_length = []
        all_metric = []
        list_dir = os.listdir(result_dir)
        new_list_dir = []
        for _item in list_dir:
            dir_path = os.path.join(result_dir, _item)
            if os.path.isdir(dir_path) and os.listdir(dir_path):
                new_list_dir.append(_item)
            else:
                continue

        for dir in new_list_dir:
            length = 0
            dir_path = os.path.join(result_dir, dir)
            json_list = get_all_json_files(dir_path)
            for json_file in json_list:
                data = read_file(json_file)
                all_response = data["reasoning"]
                for res in all_response:
                    length += len(tokenizer.encode(res))

            # csv_path = os.path.join(dir_path, "resp", "Results.csv")
            csv_path = os.path.join(dir_path, "Results.csv")
            csv_data = read_file(csv_path)
            subtasks_list = csv_data["subtask"].tolist()
            break

            

        sorted_indices = sorted(range(len(new_list_dir)), key=lambda i: float(new_list_dir[i].split('_')[-1]))
        for subtask in subtasks_list:
            all_length = []
            all_true_length = []
            all_false_length = []
            all_metric = []
            all_rea_fidelity = []
            all_res_fidelity = []
            for dir in new_list_dir:
                length = 0
                true_length = 0
                false_length = 0
                num = 0
                num_true = 0
                num_false = 0
                dir_path = os.path.join(result_dir, dir)
                json_list = get_all_json_files(dir_path)

                for json_file in json_list:
                    filename = os.path.basename(json_file)
                    filename, _ = os.path.splitext(filename)
                    if filename == subtask:
                        data = read_file(json_file)
                        all_response = data["reasoning"]
                        all_scores = data["score"]
                        # for res, score in zip(all_response, all_scores):
                        #     # length += len(tokenizer.encode(res))
                        #     num += 1
                        #     if score == 1:
                        #         true_length += len(tokenizer.encode(res))
                        #         num_true += 1
                        #     else:
                        #         false_length += len(tokenizer.encode(res))
                        #         num_false += 1

                csv_path = os.path.join(dir_path, "Results.csv")
                csv_data = read_file(csv_path)
                csv_data = csv_data[csv_data["subtask"] == subtask]
            
                metric = csv_data["score"].tolist()[-1]
                # all_length.append(length/num)

                # if num_true == 0:
                #     for json_file in json_list:
                #         all_true_length.append(0)
                #         all_false_length.append(0)
                # else:
                #     all_true_length.append(true_length/num_true)
                #     all_false_length.append(false_length/num_false)
                all_metric.append(metric)
                all_rea_fidelity.append(csv_data["rea_fidelity"].tolist()[-1])
                all_res_fidelity.append(csv_data["rep_fidelity"].tolist()[-1])

            sorted_list_dir = [new_list_dir[i] for i in sorted_indices]
            # sorted_all_length = [all_length[i] for i in sorted_indices]
            sorted_all_metric = [all_metric[i] for i in sorted_indices]
            # sorted_all_true_length = [all_true_length[i] for i in sorted_indices]
            # sorted_all_false_length = [all_false_length[i] for i in sorted_indices]
            sorted_all_rea_fidelity = [all_rea_fidelity[i] for i in sorted_indices]
            sorted_all_res_fidelity = [all_res_fidelity[i] for i in sorted_indices]

            lists_to_excel4(sorted_list_dir, sorted_all_metric, sorted_all_rea_fidelity, sorted_all_res_fidelity, filename=os.path.join(result_dir, f'{subtask}.xlsx'), column_names=("Strength" ,'Metric', 'rea_fidelity', 'res_fidelity'))



    folder_path = result_dir
    metric_df = pd.DataFrame()
    length_df = pd.DataFrame()

    list_dir = os.listdir(folder_path)
    list_dir = [item for item in list_dir if item != "FINAL.xlsx"] + ["FINAL.xlsx"]
    for filename in list_dir:
        if filename.endswith('.xlsx') and "metric_summary" not in filename and "length_summary" not in filename:
            file_path = os.path.join(folder_path, filename)

            df = pd.read_excel(file_path)

            if {'Strength', 'Metric',}.issubset(df.columns):
                df.set_index('Strength', inplace=True)
                col_name = os.path.splitext(filename)[0]
                metric_df[col_name] = df['Metric']
                metric_df[col_name + "_rea_fidelity"] = df['rea_fidelity']
                metric_df[col_name + "_rep_fidelity"] = df['res_fidelity']
                
                
                # length_df[col_name] = df['Length']
            else:
                continue

    metric_df.to_excel(os.path.join(result_dir, 'metric_summary.xlsx'))
    length_df.to_excel(os.path.join(result_dir, 'length_summary.xlsx'))
    print("save to ", os.path.join(result_dir, 'metric_summary.xlsx'))


    
if __name__ == "__main__":
    fire.Fire(main)
