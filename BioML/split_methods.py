from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, Generator
from sklearn.model_selection import GroupKFold


class CustomSplitter(Protocol):

    n_splits: int
    test_size: int | float

    def split(self, X, y, groups=None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        A function that returns train and test indices for each fold
        """
        ...

    def get_n_splits(self, X, y, groups=None) -> int:
        """
        A function that returns the number of splits
        """
        ...


@dataclass(slots=True)
class ShuffleGroupKFold:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int | None = None

    @staticmethod
    def get_sample_size(test_size:int | float, X: pd.DataFrame | np.ndarray):
        match test_size:
            case float(val) if 1 > val > 0 :
                num_test = int(val*X.shape[0])
            case int(val) if val < X.shape[0]:
                num_test = val
            case _:
                raise TypeError("test_size must be an integer less than the sample size or a float between 0 and 1")

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

    def split(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None=None, 
              groups=None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        train_data = X.copy()
        train_group = np.array(groups).copy()
        group_kfold = GroupKFold(n_splits=self.n_splits)
        if self.shuffle:
            train_data, train_group = shuffle(train_data, train_group, random_state=self.random_state)
        for i, (train_index, test_index) in enumerate(group_kfold.split(train_data, train_group, groups=groups)):
            yield train_index, test_index
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None=None, 
                         test_size:int | float = 0.2, groups=None):
        
        num_test = self.get_sample_size(test_size, X)
        train_group = np.array(groups).copy()
        train_data = X.copy()
        if self.shuffle:
            train_data, train_group = shuffle(train_data, train_group, random_state=self.random_state)
        # generate the train_test_split
        group_kfold = GroupKFold(n_splits=X.shape[0]//num_test)
        for i, (train_index, test_index) in enumerate(group_kfold.split(train_data, y, groups=train_group)):
            X_train, X_test = self.match_type(train_data, train_index, test_index)
            if y is not None:
                y_train, y_test = self.match_type(y, train_index, test_index)
                return X_train, X_test, y_train, y_test
            return X_train, X_test
    
    def get_n_splits(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None=None, 
                     groups=None) -> int:
        return self.n_splits


@dataclass
class ClusterSpliter:
    cluster_info: dict[str | int, list[str|int]]
    num_splits: int = 5
    shuffle: bool = True
    random_state: int | None = None

    @property
    def group_kfold(self):
        group = ShuffleGroupKFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.random_state)
        return group
    
    @cached_property
    def group_index(self):
        return {inde: num for num, inde in enumerate(self.cluster_info.keys())}
    
    @cached_property
    def cluster_group(self):
        return {self.group_index[x]: v for x, v in self.cluster_info.items()}

    def get_group_index(self, X: pd.DataFrame):
        group = []
        for x in X.index:
            for key, value in self.cluster_group.items():
                if x in value:
                    group.append(key)
                    break
        group = np.array(group)
        return group

    def train_test_split(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None=None, 
                         test_size:int | float = 0.2):
        
        return self.group_kfold.train_test_split(X, y, test_size=test_size, groups=self.get_group_index(X))


    def split(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None=None, 
              groups=None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        
        group = self.get_group_index(X)
        for train_index, test_index in self.group_kfold.split(X, y, groups=group):
            yield train_index, test_index
    
    def get_n_splits(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None=None, 
                     groups=None) -> int:
        return self.num_splits




