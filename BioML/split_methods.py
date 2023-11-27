import time
import random
from sklearn.utils import shuffle
import numpy as np
import warnings
import pandas as pd
from dataclasses import dataclass, field
from .custom_errors import InsufficientSamplesError, InsufficientClustersError


@dataclass(slots=True)
class ClusterSpliter:
    cluster_info: dict[str | int, list[str|int]]
    num_splits: int = 5
    random_state: int | None = 10
    shuffle: bool = True
    done: list = field(default_factory=list)
    done_copy: list = field(default_factory=list)

    def __post_init__(self):
        if len(self.cluster_info) < self.num_splits:
            raise InsufficientClustersError(f"The number of clusters is less than the number of folds. " 
                                            f"Use {len(self.cluster_info)} or less folds instead, or "
                                            "increase the identity threshold. I recommend the former")

    def get_test_index(self, index, num_test) -> np.ndarray:
        start = time.perf_counter()
        test_ind: list = []

        if len(index) <= num_test: raise InsufficientSamplesError("The number of samples is less or equal than the "
                                                                  "defined test size")

        if self.random_state is not None: random.seed(self.random_state)
        while len(test_ind) < num_test:
            x = random.choice(list(self.cluster_info.keys()))
            if x not in self.done: self.done.append(x)
            else: continue
            # handle when it is more, I will wait 1.4 minutes and then proceed by selecting the samples
            end = time.perf_counter()
            if len(test_ind) + len(self.cluster_info[x]) > int(num_test*1.2):
                # if it is more than 1.4 minutes, then I will proceed by selecting the samples
                if (end - start) / 60 < 1.4: 
                    self.done.remove(x)
                    continue
                # warns the user that the number of samples in the test set is more than the defined number
                warnings.warn(f"The number of samples in the test set is more than the defined number, "
                              "could not find a cluster with a suitable size " 
                            f"{len(test_ind) + len(self.cluster_info[x])} instead of {num_test}")

            test_ind.extend(self.cluster_info[x])

        test = np.array([index[x] for x in test_ind])
        if self.shuffle:
            test =  shuffle(test, random_state=self.random_state)

        return test
    
    def clear(self):
        self.done.clear()

    def get_train_index(self, test_ind, index):
        train = np.array([x for x in index.values() if x not in test_ind])
        if self.shuffle:
            train = shuffle(train, random_state=self.random_state)
        return train

    def handle_less_samples(self, num_test, index, test_dict, train_dict, split_ind, discard_fold=0.7) -> bool:
        # handle cases when what it is left is less than the number of defined test samples before the last fold
        break_ = False
        left = sum(len(v) for x, v in self.cluster_info.items() if x not in self.done)

        if left < int(num_test*discard_fold):

            warnings.warn(f"The number of samples left for the test set in the next fold is much less than the defined number: "
                    f"{left} instead of {num_test} samples for fold {split_ind+2}, of the total of {self.num_splits} folds, "
                    "decrease the number of folds or the test set size. The loop will stop and the next fold will be discarded."
                    "if you want to mantain the number of folds, decrease the required test size."
                    "if you want to mantain the test size, decrease the number of folds.")
            
            break_ = True

        elif int(num_test*1.1) > left >= int(num_test*discard_fold):
            warnings.warn(f"The number of samples left for the test set is only sufficient for the next fold: "
                    f"{left} instead of {num_test} samples for fold {split_ind+2}, of the total of {self.num_splits} folds. "
                    "It will use these samples for the next fold and stop. "
                    "if you want to mantain the number of folds, decrease the required test size. "
                    "if you want to mantain the test size, decrease the number of folds.")
            
            test = np.array([index[y] for key, value in self.cluster_info.items() for y in value 
                                            if key not in self.done])
            if self.shuffle:
                test = shuffle(test, random_state=self.random_state)

            test_dict[split_ind+2] = test
            train_dict[split_ind+2] = self.get_train_index(test, index)
            break_ = True

        return break_
    
    @staticmethod
    def get_sample_size(test_size:int | float, X: pd.DataFrame | np.ndarray | None=None):
        match test_size:
            case float(val) if 1 > val > 0 :
                num_test = int(val*X.shape[0])
            case int(val):
                num_test = val
            case _:
                raise TypeError("test_size must be an integer or a float between 0 and 1")

        return num_test
    
    @staticmethod
    def match_type(data, train_index, test_index):
        match data:
            case pd.DataFrame() | pd.Series() as val:
                return val.iloc[train_index], val.iloc[test_index]
            case np.ndarray() as val:
                return val[train_index], val[test_index]
            case _:
                raise TypeError("X must be a pandas DataFrame or a numpy array")

    @staticmethod
    def get_sample_names(X):
        match X:
            case pd.DataFrame() | pd.Series() as val:
                return val.index
            case np.ndarray() as val:
                return range(val.shape[0])
            case _:
                raise TypeError("X must be a pandas DataFrame, Series or a numpy array")

    def train_test_split(self, X: pd.DataFrame | np.ndarray, y=None, test_size:int | float = 0.2, clear=True):
        
        # clear the done list each time the function is called
        if clear: self.clear()
        
        sample_names = self.get_sample_names(X)
        print(sample_names)
        index = {inde: num for num, inde in enumerate(sample_names)}
        
        num_test = self.get_sample_size(test_size, X)

        test_index = self.get_test_index(index, num_test)
        train_index = self.get_train_index(test_index, index)

        X_train, X_test = self.match_type(X, train_index, test_index)
        self.done_copy = self.done.copy()
        
        if y is not None:
            y_train, y_test = self.match_type(y, train_index, test_index)

            return X_train, X_test, y_train, y_test

        return X_train, X_test

    def get_fold_indices(self, X: pd.DataFrame | np.ndarray, test_size:int | float = 0.2, 
                         discard_factor=0.7):
        
        self.done = self.done_copy.copy()
        num_test = min(X.shape[0] // self.num_splits, self.get_sample_size(test_size, X))
        sample_names = self.get_sample_names(X)
        index = {inde: num for num, inde in enumerate(sample_names)}
        test = {}
        train = {}
        for ind in range(self.num_splits):
            test[ind+1] = self.get_test_index(index, num_test)
            train[ind+1] = self.get_train_index(test[ind+1], index)
            break_ = self.handle_less_samples(num_test, index, test, train, ind, discard_factor)

            if break_: break

        return train, test


class CustomSplitter:
    def __init__(self, spliting_fn, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size
        self.spliting_fn = spliting_fn
    
    def split(self, X, y, groups=None):
        train, test = self.spliting_fn(X, self.test_size)
        for ind in test:
            yield (train[ind], test[ind])

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits