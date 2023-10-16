from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import ast
import numpy as np
import logging
from collections import Counter


def scale(scaler, X_train, X_test=None):
    """
    Scale the features using RobustScaler
    """
    scaler_dict = {"robust": RobustScaler(), "zscore": StandardScaler(), "minmax": MinMaxScaler()}
    transformed = scaler_dict[scaler].fit_transform(X_train)
    #transformed = pd.DataFrame(transformed, index=X_train.index, columns=X_train.columns)
    if X_test is None:
        return transformed, scaler_dict
    else:
        test_x = scaler_dict[scaler].transform(X_test)
        #test_x = pd.DataFrame(test_x, index=X_test.index, columns=X_test.columns)
        return transformed, scaler_dict, test_x


def write_excel(file, dataframe, sheet_name, overwrite=False):
    if not isinstance(file, pd.io.excel._openpyxl.OpenpyxlWriter):
        if overwrite or not Path(file).exists():
            mode = "w"
        else:
            mode = "a"
        with pd.ExcelWriter(file, mode=mode, engine="openpyxl") as writer:
            dataframe.to_excel(writer, sheet_name=sheet_name)
    else:
        dataframe.to_excel(file, sheet_name=sheet_name)


def interesting_classifiers(name, params):
    """
    All classifiers
    """
    classifiers = {
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "SGDClassifier": SGDClassifier,
        "RidgeClassifier": RidgeClassifier,
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
        "MLPClassifier": MLPClassifier,
        "SVC": SVC,
        "XGBClassifier": XGBClassifier,
        "LGBMClassifier": LGBMClassifier,
        "KNeighborsClassifier": KNeighborsClassifier
    }

    return classifiers[name](**params)


def modify_param(param, name, num_threads=-1):
    if "n_jobs" in param:
        param["n_jobs"] = num_threads
    if "shuffle" in param:
        del param["shuffle"]
    if "random_state" in param and param["random_state"] == True:
        param["random_state"] = 1
    if "fit_intercept" in param:
        del param["fit_intercept"]
    if "MLPClassifier" in name:
        param['hidden_layer_sizes'] = ast.literal_eval(param['hidden_layer_sizes'])
    if "XGBClassifier" in name:
        if param['scale_pos_weight'] == True:
            param['scale_pos_weight'] = 1
        if param['scale_pos_weight'] == False:
            param['scale_pos_weight'] = 0
        if param['missing'] is None:
            param["missing"] = np.nan
        if param["max_delta_step"] == False:
            param["max_delta_step"] = None
    if "LGBMClassifier" in name:
        if param["min_split_gain"] == False:
            param["min_split_gain"] = 0.0
        if param["min_split_gain"] == True:
            param["min_split_gain"] = 1.0
        if param["subsample_freq"] == False:
            param["subsample_freq"] = 0
        if param["subsample_freq"] == True:
            param["subsample_freq"] = 1
        if param["max_delta_step"] == False:
            param["max_delta_step"] = None
    if "RidgeClassifier" in name:
        del param["normalize"]
    if "MLPClassifier" in name:
        if param['nesterovs_momentum'] == 1:
            param['nesterovs_momentum'] = True
    if "SVC" in name:
        if "shrinking" in param:
            if param["shrinking"] == 1:
                param["shrinking"] = True
            if param["shrinking"] == 0:
                param["shrinking"] = False
    return param


def rewrite_possum(possum_stand_alone_path):
    possum_path = Path(possum_stand_alone_path)
    with possum_path.open() as possum:
        possum = possum.readlines()
        new_possum = []
        for line in possum:
            if "python" in line:
                new_line = line.split(" ")
                if "possum.py" in line:
                    new_line[2] = f"{possum_path.parent}/src/possum.py"
                else:
                    new_line[2] = f"{possum_path.parent}/src/headerHandler.py"
                line = " ".join(new_line)
                new_possum.append(line)
            else:
                new_possum.append(line)
    with open(possum_path, "w") as possum_out:
        possum_out.writelines(new_possum)

class Log:
    """
    A class to keep log of the output from different modules
    """

    def __init__(self, name):
        """
        Initialize the Log class

        Parameters
        __________
        name: str
            The name of the log file
        """
        self._logger = logging.getLogger(name)
        self._logger.handlers = []
        self._logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler("{}.log".format(name))
        self.fh.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%d-%m-%Y %H:%M:%S")
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        self._logger.addHandler(self.fh)
        self._logger.addHandler(self.ch)

    def debug(self, messages, exc_info=False):
        """
        It pulls a debug message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message            
        """
        self._logger.debug(messages, exc_info=exc_info)

    def info(self, messages, exc_info=False):
        """
        It pulls an info message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message            
        """
        self._logger.info(messages, exc_info=exc_info)

    def warning(self, messages, exc_info=False):
        """
        It pulls a warning message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message            
        """
        self._logger.warning(messages, exc_info=exc_info)

    def error(self, messages, exc_info=False):
        """
        It pulls a error message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message
        """
        self._logger.error(messages, exc_info=exc_info)

    def critical(self, messages, exc_info=False):
        """
        It pulls a critical message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message
        """
        self._logger.critical(messages, exc_info=exc_info)


class Threshold:

    def __init__(self, csv_file, output_csv, sep=","):
        self.csv_file = pd.read_csv(csv_file, sep=sep, index_col=0)
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(exist_ok=True, parents=True)

    def apply_threshold(self, threshold, greater=True, column_name='temperature'):
        """
        Apply threshold to dataset
        :param threshold: threshold value
        :param greater: boolean value to indicate if threshold is greater or lower
        :param column_name: column name to apply threshold
        :return: filtered dataset
        """
        dataset = self.csv_file[column_name].copy()
        dataset = pd.to_numeric(dataset, errors="coerce")
        dataset.dropna(inplace=True)
        data = dataset.copy()
        if greater:
            data.loc[dataset >= threshold] = 1
            data.loc[dataset < threshold] = 0
        else:
            data.loc[dataset <= threshold] = 1
            data.loc[dataset > threshold] = 0
        print(f"using the threshold {threshold} returns these proportions", Counter(data))
        return data

    def save_csv(self, threshold, greater=True, column_name='temperature'):
        data = self.apply_threshold(threshold, greater, column_name)
        data.to_csv(self.output_csv)