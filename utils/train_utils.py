import os
import json
import pandas as pd

class ExperimentManager(object):
    def __init__(self, exp_id, model_name, repeats=1):
        cwd = os.path.abspath(os.path.dirname(__file__))
        self._exp_folder_path = os.path.abspath(os.path.join(cwd, r"../experiments", exp_id))
        self._results_folder_path = os.path.join(self._exp_folder_path, r"results")
        self._checkpoints_folder_path = os.path.join(self._exp_folder_path, r"checkpoints")
        self._models_folder_path = os.path.join(self._exp_folder_path, r"models")
        self._repeats = repeats
        self._current_repeat = 0
        self._model_name = model_name
        self._eval = {"t": []}
        os.makedirs(self._exp_folder_path, exist_ok=True)
        os.makedirs(self._results_folder_path, exist_ok=True)
        os.makedirs(self._checkpoints_folder_path, exist_ok=True)
        os.makedirs(self._models_folder_path, exist_ok=True)
        if repeats > 1:
            for i in range(repeats):
                os.makedirs(os.path.join(self._results_folder_path, str(i)))
                os.makedirs(os.path.join(self._checkpoints_folder_path, str(i)))
                os.makedirs(os.path.join(self._models_folder_path, str(i)))


    def save_parameters_json(self, parameter_dict, name):
        with open(os.path.join(self._exp_folder_path, name), "w") as fp:
            json.dump(parameter_dict, fp) 
        
    def set_policy(self, policy):
        self.policy = policy

    def end_training(self):
        if self._repeats > 1:
            self.policy.save_model(os.path.join(self._models_folder_path, str(self._current_repeat), self._model_name))
            result_path = os.path.join(self._results_folder_path, str(self._current_repeat))
            self._current_repeat += 1
        else:
            self.policy.save_model(os.path.join(self._models_folder_path, self._model_name))
            result_path = self._results_folder_path
        eval_df = pd.DataFrame.from_dict(self._eval)
        eval_df.to_csv(os.path.join(result_path, "eval.csv"))
        self._eval = {"t": []}

    def eval(self, results, t):
        self._eval["t"].append(t)
        for key, value in results.items():
            if key in self._eval.keys():
                self._eval[key].append(value)
            else:
                self._eval[key] = [value]