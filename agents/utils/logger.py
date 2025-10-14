import datetime
import errno
import numpy as np
import pandas as pd
import gc
import os
import json
from numpy.typing import NDArray
from typing import Dict, List

OUTPUT_FILE_EXTENSION = ".csv"
WEIGHTS_FILE_EXTENSION = ".npy"
PARAMS_FILE_EXTENSION = ".txt"

class Logger:
    def __init__(
        self,
        curdir: str,
        params: Dict,
        l_params: Dict,
        log_params: Dict,
        train: bool,
        buffer_size: int,
        weights_file=None
    ) -> None:

        mode = "train" if train else "eval"
        time_now = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

        base_dir = os.path.join(curdir, "runs/" + mode)
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)

        output_dir = os.path.join(base_dir, mode + "_" + time_now)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        filename = log_params[mode + "_output_file"].replace("-", "_") + '_' + time_now + OUTPUT_FILE_EXTENSION
        self.output_file = os.path.join(output_dir, filename)
        
        params_filename = log_params[mode + "_params_file"] + '_' + time_now + PARAMS_FILE_EXTENSION
        self.params_file = os.path.join(output_dir, params_filename)
        
        weights_dir = os.path.join(curdir, "runs/weights")
        if not os.path.isdir(weights_dir):
            os.makedirs(weights_dir)
        
        if train:
            weights_filename = log_params[mode + "_weights_file"] + '_' + time_now + WEIGHTS_FILE_EXTENSION 
            self.weights_file = os.path.join(weights_dir, weights_filename)
        else:
            if weights_file is None:
                if not os.path.isdir(weights_dir):
                    raise FileNotFoundError(errno.ENOENT, "No such directory", weights_dir)
                elif len(os.listdir(weights_dir)) == 0:
                    raise FileNotFoundError(errno.ENOENT, "This directory is empty", weights_dir)
                    
                self.weights_file = self._get_weight_path(weights_dir, WEIGHTS_FILE_EXTENSION)
            else:
                self.weights_file = weights_file

        self._write_params(params, l_params, log_params, train)
        self.metrics = tuple(self._get_metrics(params, train))
        self.table = pd.DataFrame(columns=self.metrics)
        self.buffer_size = buffer_size
    
    def _get_weight_path(self, weights_dir: str, ext: str) -> str:
        """
        Return qtable weights file path.
        """
        
        weights_filename = sorted([
            f for f in os.listdir(weights_dir) if os.path.isfile(os.path.join(weights_dir, f)) and f.endswith(ext)
        ])[-1]        
        if len(weights_filename) == 0:
            raise FileNotFoundError(errno.ENOENT, f"No such weights file ({WEIGHTS_FILE_EXTENSION}) found in", weights_dir)
        weights_file = os.path.join(weights_dir, weights_filename)
        
        return weights_file

    def _write_params(
        self,
        params: Dict,
        l_params: Dict,
        log_params: Dict,
        train: bool
    ) -> None:
        """
        Write to file all parametrs used in training/eval.
        """

        with open(self.params_file, 'w') as f:
            f.write(f"{json.dumps(params, indent=2)}\n")
            f.write("----------\n")
            f.write(f"{json.dumps(l_params, indent=2)}\n")
            f.write("----------\n")
            f.write(f"{json.dumps(log_params, indent=2)}\n")
            if not train:
                f.write(f"weights_file = {self.weights_file}\n")
            f.write("----------\n")
            
    def _get_metrics(self, params: dict, train: bool) -> List:
        """
        Return all logged metrics. 
        """
        
        metrics = ["Episode", "Tick"]

        if params["cluster_learners"] == 0 or params["scatter_learners"] == 0:
            metrics.append("Avg cluster X episode")
        else:
            double_agent_metrics = [
                "Avg only cluster X episode",
                "Avg mixed cluster X episode",
                "Avg only scatter X episode",
                "Avg mixed scatter X episode"
            ]
            metrics.extend(double_agent_metrics)

        # Clustering
        if params["cluster_learners"] > 0:
            metrics.append("Cluster avg reward X episode") 
            for a in params["actions"]:
                metrics.append("Cluster " + a)
            for l in range(params["cluster_learners"]):
                for a in params["actions"]:
                    metrics.append(f"(cluster_learner {l})-{a}")

        # Scattering
        if params["scatter_learners"] > 0:
            metrics.append("Scatter avg reward X episode") 
            for a in params["actions"]:
                metrics.append("Scatter " + a)
            for l in range(params["scatter_learners"]):
                for a in params["actions"]:
                    metrics.append(f"(scatter_learner {l})-{a}")

        if train:
            metrics.append("Epsilon")

        return metrics
    
    def _write_to_csv(self) -> bool:
        """
        Write metrics to the csv file.
        """

        if os.path.isfile(self.output_file): # check se il file esiste
            with open(self.output_file, 'a') as f:
                self.table.to_csv(f, header=False, sep=',', index=False)
        else: # check se il file non esiste
            self.table.to_csv(self.output_file, sep=',', index=False)
        return True

    def _delete_table(self) -> None:
        """
        Delete logging table.
        """

        self.table = None
        gc.collect()

    def _reinit(self) -> None:
        """
        Re-initialize the metrics logging table.
        """

        self._delete_table()
        self.table = pd.DataFrame(columns=self.metrics)

    def _add_rows(self, vals: List) -> None:
        """
        Add a row to the logging table.
        """

        tmp = pd.DataFrame(vals, columns=self.metrics)
        if self.table.shape[0] == 0:
            self.table = self.table.combine_first(tmp)
        else:
            self.table = pd.concat([self.table, tmp], ignore_index=True)

    def load_value(self, value: List) -> None:
        """
        Load a value to the logging table.
        """

        assert(isinstance(value, list) or isinstance(value, tuple)), "Error: value must be of type list or tuple!"
        quantity = self.buffer_size - self.table.shape[0]
        if quantity == 0:
            flag = self._write_to_csv()
            if flag:
                self._reinit()
        self._add_rows([value])

    def empty_table(self) -> None:
        """
        Empty logging table.
        """

        if self.table.shape[0] > 0:
            self._write_to_csv()
        self._delete_table()

    def save_model(self, weights: NDArray):
        """
        Save the qtable.
        """

        np.save(self.weights_file, weights)

    def load_model(self) -> NDArray:
        """
        Load the qtable.
        """

        return np.load(self.weights_file)    
    
    def save_computation_time(
        self,
        computation_time,
        train=True
    ) -> None:
        """
        Write computation time to file.
        """ 

        with open(self.params_file, 'a') as f:
            if train:
                f.write(f"Training time: {computation_time}\n")
            else:
                f.write(f"Testing time: {computation_time}\n")