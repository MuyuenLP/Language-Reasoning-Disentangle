import glob, os
current_path = os.getcwd()

from freeEvalLM._lib._df import read_file


class TaskLoader:
    def __init__(self, 
        task, 
        save_path,
        sample: int = -1
        ):
         self.task = task
         self.save_path = save_path
         self.sample = sample
         self.doc()
         
    def doc(self):
        # print(self.task)
        if self.task == "mgsm":
            self.data_path = os.path.join("datasets/mgsm_0shots")
        elif self.task == "mmmlu_for_mlrs":
            self.data_path = os.path.join("datasets/mmmlu_for_mlrs")
        elif self.task == "xwinograd_for_mlrs":
            self.data_path = os.path.join("datasets/xwinograd_for_mlrs")
        else:
            print("self.task is", self.task)


         
    def load_dataset(self):
        print("loading dataset")
        print(self.data_path,)
        subtasks_data_path = glob.glob(os.path.join(self.data_path, "*.json")) + glob.glob(os.path.join(self.data_path, "*.csv"))
        all_dfs = []
        print(subtasks_data_path)

        for data_path in subtasks_data_path:
            if  os.path.basename(data_path).split(".")[0] == "Results":
                continue
            print("loading subtask: ", os.path.basename(data_path).split(".")[0])
            subtask_name = os.path.basename(data_path).split(".")[0]
            subtask_data = read_file(data_path)

            if self.sample > 0:
                subtask_data = subtask_data.head(self.sample).reset_index(drop=True)

            _dict = {
                "subtask_name": subtask_name,
                "subtask_data": subtask_data
            }

            all_dfs.append(_dict)
        
        return all_dfs
            
    def load_evaluator(self):
        if self.task == "mgsm":
            from freeEvalLM.tasks.mgsm.mgsm import mgsm as Evaluator
        elif self.task == "mmmlu_for_mlrs":
            from freeEvalLM.tasks.mmmlu.mmmlu import mmmlu as Evaluator
        elif self.task == "xwinograd_for_mlrs":
            from freeEvalLM.tasks.xwinograd.xwinograd import xwinograd as Evaluator
        else:
            from freeEvalLM.src.Evaluator import Evaluator as Evaluator

        evaluator = Evaluator(self.task, self.save_path)
        return evaluator


if __name__ == "__main__":
    a = TaskLoader("mgsm", "test_results")
    a.load_dataset()






