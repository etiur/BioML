from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, Generator, Iterable
from sklearn.model_selection import GroupKFold
import operator


def match_type(data: pd.DataFrame | pd.Series | np.ndarray,
               train_index: list[int] | np.ndarray, test_index: list[int] | np.ndarray):
    """
    A function to match the type of the data and return the train and test sets.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series | np.ndarray
        The data
    train_index : list[int] | np.ndarray
        The train index
    test_index : list[int] | np.ndarray
        The test index

    Returns
    -------
    pd.DataFrame | pd.Series | np.ndarray
        The train and test sets

    Raises
    ------
    TypeError
        data must be a pandas DataFrame or a numpy array
    """
    match data:
        case pd.DataFrame() | pd.Series() as val:
            return val.iloc[train_index], val.iloc[test_index]
        case np.ndarray() as val:
            return val[train_index], val[test_index]
        case _:
            raise TypeError("data must be a pandas DataFrame or a numpy array")


class CustomSplitter(Protocol):
    """
    An structural subtyping class (like an abstract class). 
    """
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
    
    def train_test_split(self, X, y, test_size:int | float = 0.2, groups=None):
        """
        A function that returns the train and test sets
        """
        ...


@dataclass(slots=True)
class ShuffleGroupKFold:
    """
    A class to split data based on groups. 
    A wrapper around the GroupKFold class in scikit-learn that shuffle the groups.

    Parameters
    ----------
    n_splits : int, optional
        The number of splits, by default 5
    shuffle : bool, optional
        Shuffle the data, by default True
    random_state : int, optional
        Random state, by default None

    """
    n_splits: int = 5
    shuffle: bool = True
    random_state: int | None = None

    @staticmethod
    def get_sample_size(test_size:int | float, X: pd.DataFrame | np.ndarray):
        """
        Get the sample size for the test set.

        Parameters
        ----------
        test_size : int | float
            The size of the test set
        X : pd.DataFrame | np.ndarray
            The data

        Returns
        -------
        int
            The sample size for the test set

        Raises
        ------
        TypeError
            test_size must be an integer less than the sample size or a float between 0 and 1
        """
        match test_size:
            case float(val) if 1 > val > 0 :
                num_test = int(val*X.shape[0])
            case int(val) if val < X.shape[0]:
                num_test = val
            case _:
                raise TypeError("test_size must be an integer less than the sample size or a float between 0 and 1")

        return num_test

    def split(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None=None, 
              groups: Iterable[str|int] =None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split the data into train and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            The data
        y : pd.Series | np.ndarray | None, optional
            The target variable, by default None
        groups : Iterable[str|int], optional
            The groups, by default None

        Yields
        ------
        Generator[tuple[np.ndarray, np.ndarray], None, None]
            The train and test indices for each fold
        """
        train_data = X.copy()
        train_group = np.array(groups).copy()
        group_kfold = GroupKFold(n_splits=self.n_splits)
        if self.shuffle:
            train_data, train_group = shuffle(train_data, train_group, random_state=self.random_state)
        for i, (train_index, test_index) in enumerate(group_kfold.split(train_data, y, groups=train_group)):
            yield train_index, test_index
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None=None, 
                         test_size:int | float = 0.2, groups=None):
        """
        Split the data into train and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            The data
        y : pd.Series | np.ndarray | None, optional
            The labels, by default None
        test_size : int | float, optional
            The test size in int or float, by default 0.2
        groups : Iterable[int, str], optional
            The group  for each sample, by default None

        Returns
        -------
        pd.DataFrame | pd.Series | np.ndarray
            The train and test sets
            
        """
        num_test = self.get_sample_size(test_size, X)
        train_group = np.array(groups).copy()
        train_data = X.copy()
        if self.shuffle:
            train_data, train_group = shuffle(train_data, train_group, random_state=self.random_state)
        # generate the train_test_split
        group_kfold = GroupKFold(n_splits=X.shape[0]//num_test)
        for i, (train_index, test_index) in enumerate(group_kfold.split(train_data, y, groups=train_group)):
            X_train, X_test = match_type(train_data, train_index, test_index)
            if y is not None:
                y_train, y_test = match_type(y, train_index, test_index)
                return X_train, X_test, y_train, y_test
            return X_train, X_test
    
    def get_n_splits(self, X: pd.DataFrame | np.ndarray | None= None, y: pd.Series | np.ndarray | None=None, 
                     groups=None) -> int:
        return self.n_splits


@dataclass
class ClusterSpliter:
    """
    A class to split data based on clusters.

    Parameters
    ----------
    _cluster_info : dict[str | int, list[str|int]] | str
        A dictionary containing the cluster information or a file path to the cluster information. 
        The dictionary should have the cluster identifier as the key and the list of samples in the cluster as the value.
    num_splits : int, optional
        The number of splits, by default 5
    shuffle : bool, optional
        Shuffle the data, by default True
    random_state : int, optional
        Random state, by default None
    test_size : int | float, optional
        The size of the test set, by default 0.2
    """
    _cluster_info: dict[str | int, list[str|int]] | str
    num_splits: int = 5
    shuffle: bool = True
    random_state: int | None = None
    test_size:int | float = 0.2

    @property
    def cluster_info(self):
        """
        Get the cluster information.

        Returns
        -------
        dict[str | int, list[str|int]]
            A dictionary containing the cluster information.
        """
        if isinstance(self._cluster_info, str):
            return self.read_cluster_info(self._cluster_info)
        return self._cluster_info

    @cached_property
    def group_kfold(self):
        """
        Get the group kfold object.

        Returns
        -------
        ShuffleGroupKFold
            The group kfold object which is the general object for splitting the data based on a list of groups
        """
        group = ShuffleGroupKFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.random_state)
        return group
    
    def read_cluster_info(self, file_path: str):
        """
        Read the cluster information from a tsv file generated by MMSeqs2 function from utilities.

        Parameters
        ----------
        file_path : str
            The file path to the cluster information.

        Returns
        -------
        dict[str | int, list[str|int]]
            A dictionary containing the cluster information.
        """
        cluster_info = {}
        with open(file_path, "r") as f:
            lines = [x.strip() for x in f.readlines()]
        for x in lines:
            X = x.split("\t")
            if X[0] not in cluster_info:
                cluster_info[X[0]] = []
            cluster_info[X[0]].append(X[1])
        return cluster_info

    def get_group_index(self, X: pd.DataFrame):
        """
        Now for each sample in the data, assign a group using the cluster info dictionary.
        The groups will be the keys of the cluster info.

        X.index should be the same as the cluster_info values.

        Parameters
        ----------
        X : pd.DataFrame
            The data

        Returns
        -------
        np.ndarray
            The group index for the data.
        """
        group = []
        for x in X.index:
            for key, value in self.cluster_info.items():
                if x in value:
                    group.append(key)
                    break
        group = np.array(group)
        return group

    def train_test_split(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None=None, 
                         groups=None):
        """
        Split the data into train and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            
        y : pd.Series | np.ndarray | None, optional
            _description_, by default None
        groups : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        return self.group_kfold.train_test_split(X, y, test_size=self.test_size, 
                                                 groups=self.get_group_index(X))


    def split(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None=None, 
              groups=None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        
        group = self.get_group_index(X)
        for train_index, test_index in self.group_kfold.split(X, y, groups=group):
            yield train_index, test_index
    
    def get_n_splits(self, X: pd.DataFrame | np.ndarray | None= None, y: pd.Series | np.ndarray | None=None, 
                     groups=None) -> int:
        return self.num_splits


@dataclass
class MutationSpliter:
    """
    A class to split data based on the number of mutations.

    Parameters
    ----------
    mutations : np.ndarray | list[int] | str | tuple[int, ...]
        The number of mutations for each sample.
    test_num_mutations : int
        The number of mutations used as threshold for the sampel to be included in the test set.
    greater : bool, optional
        If True, the samples with number of mutations greater than the test_num_mutations will be included in the test set, by default True
    num_splits : int, optional
        The number of splits, by default 5
    shuffle : bool, optional
        Shuffle the data, by default True
    random_state : int | None, optional
        Random state, by default None
    
    Raises
    ------
    ValueError
        The number of samples in the data is not equal to the list of the number of mutations

    TypeError
        mutations must be an array of number of mutations or a string if the mutations are in the columns
    """
    mutations: np.ndarray | list[int] | str | tuple[int, ...]
    test_num_mutations: int
    greater: bool = True
    num_splits: int = 5
    shuffle: bool = True
    random_state: int | None = None

    @property
    def group_kfold(self):
        group = ShuffleGroupKFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.random_state)
        return group
    
    def apply_operator(self, a, b):
        if self.greater: 
            return operator.ge(a, b)
        return operator.le(a, b)

    def get_test_indices(self, mutations):
        test_indices = []
        for num, mut in enumerate(mutations):
            if self.apply_operator(mut, self.test_num_mutations):
                test_indices.append(num)
        if self.shuffle:
            test_indices =  shuffle(test_indices, random_state=self.random_state)
        return test_indices
    
    def get_train_indices(self, mutations, test_indices):
        train_indices = [num for num, x in enumerate(mutations) if num not in test_indices]
        if self.shuffle:
            train_indices = shuffle(train_indices, random_state=self.random_state)
        return train_indices

    def get_mutations(self, X: pd.DataFrame) -> tuple[pd.DataFrame, Iterable[int]]:
        """
        Get the mutations from the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data

        Returns
        -------
        tuple[pd.DataFrame, Iterable[int]]
            The mutations and the data without the mutations

        Raises
        ------
        ValueError
            The number of samples in the data is not equal to the list of the number of mutations
        TypeError
            mutations must be an array of number of mutations or a string if the mutations are in the columns of the dataframe
        """
        match self.mutations:
            case np.ndarray() | list() | tuple() as mutations:
                train_data = X.copy()
                if len(X) != len(mutations):
                    raise ValueError("The number of samples in the data is not equal to the list of the number of mutations")
                return mutations, train_data

            case str(val) if val in X.columns:
                mutations = X[val]
                train_data = X.drop(val, axis=1)
                return mutations, train_data

            case _:
                raise TypeError("mutations must be an array of number of mutations or a string if the mutations are in the columns of the dataframe")
        
    def train_test_split(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None=None, 
                         groups=None):
        """
        Split the data into train and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            The data
        y : pd.Series | np.ndarray | None, optional
            The labels, by default None
        groups : _type_, optional
            _description_, by default None

        """
        mutations, train_data = self.get_mutations(X)
            
        test_indices = self.get_test_indices(mutations)
        train_indices = self.get_train_indices(mutations, test_indices)

        X_train, X_test = match_type(train_data, train_indices, test_indices)
        if y is not None:
            y_train, y_test = match_type(y, train_indices, test_indices)
            return X_train, X_test, y_train, y_test
        return X_train, X_test
    

