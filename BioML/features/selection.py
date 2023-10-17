import argparse
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from BioML.utilities import scale, analyse_composition, write_excel, Log
from pathlib import Path
# Multiprocess instead of Multiprocessing solves the pickle problem in Windows (might be different in linux)
# but it has its own errors. Use multiprocessing.get_context('fork') seems to solve the problem but only available
# in Unix. Altough now it seems that with fork the programme can hang indefinitely so use spaw instead
# https://medium.com/devopss-hole/python-multiprocessing-pickle-issue-e2d35ccf96a9
import xgboost as xgb
from sklearn.linear_model import RidgeClassifier, Ridge
from collections import defaultdict
import numpy as np
import random
from multiprocessing import get_context  # https://pythonspeed.com/articles/python-multiprocessing/
import time
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit
from typing import Iterable
from dataclasses import dataclass
import BioML.features.methods as methods
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr


def arg_parse():
    parser = argparse.ArgumentParser(description="Preprocess and Select the best features, only use it if the feature came from possum or ifeatures")

    parser.add_argument("-f", "--features", required=False,
                        help="The path to the training features that contains both ifeature and possum in csv format",
                        default="training_features/every_features.csv")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format if not in the features,"
                             "if present in the features csv use the flag to specify the label column name")
    parser.add_argument("-r", "--feature_range", required=False, default="none:none:none",
                        help="Specify the minimum and maximum of number of features in start:stop:step format or "
                             "a single integer. Start will default to num samples / 10, Stop will default to num samples / 2 and step will be (stop - step / 5)")
    parser.add_argument("-n", "--num_thread", required=False, default=10, type=int,
                        help="The number of threads to use for parallelizing the feature selection")
    parser.add_argument("-v", "--variance_threshold", required=False, default=0, type=float,
                        help="The variance the feature has to have, 0 means that the comlun has the same value for all samples. None to deactivate")
    parser.add_argument("-s", "--scaler", required=False, default="robust", choices=("robust", "zscore", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to RobustScaler")
    parser.add_argument("-e", "--excel_file", required=False,
                        help="The file path to where the selected features will be saved in excel format",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-k", "--kfold_parameters", required=False,
                        help="The parameters for the kfold in num_split:test_size format", default="5:0.2")
    parser.add_argument("-rt", "--rfe_steps", required=False, type=int,
                        help="The number of steps for the RFE algorithm, the more step the more precise "
                             "but also more time consuming and might lead to overfitting", default=30)
    parser.add_argument("-p", "--plot", required=False, action="store_false",
                        help="Default to true, plot the feature importance using shap")
    parser.add_argument("-pk", "--plot_num_features", required=False, default=20, type=int,
                        help="How many features to include in the plot")
    parser.add_argument("-nf", "--num_filters", required=False, type=int,
                        help="The number univariate filters to use maximum 10", default=10)
    parser.add_argument("-se", "--seed", required=False, type=int, default=None,
                        help="The seed number used for reproducibility")
    parser.add_argument("-st", "--strategy", required=False, choices=("holdout", "kfold"), default="holdout",
                        help="The spliting strategy to use")
    parser.add_argument("-pr", "--problem", required=False, choices=("classification", "regression"), 
                        default="classification", help="Classification or Regression problem")

    args = parser.parse_args()

    return [args.features, args.label, args.variance_threshold, args.feature_range, args.num_thread, args.scaler,
            args.excel_file, args.kfold_parameters, args.rfe_steps, args.plot, args.plot_num_features, args.num_filters,
            args.seed, args.strategy, args.problem]

@dataclass
class DataReader:
    """
    A class for reading data from CSV files.

    Parameters
    ----------
    label : pd.Series or str or Iterable[int or float]
        The label column of the CSV file. Can be a pandas Series, a string
        representing the column name, or an iterable of integers or floats
        representing the label values.
    features : pd.DataFrame or str or list or np.ndarray, optional
        The feature columns of the CSV file. Can be a pandas DataFrame, a string
        representing the file path, a list of column names, or a numpy array
        representing the feature values.
    variance_thres : float or None, optional
        The variance threshold for feature selection. If set to a value greater
        than 0, only features with variance greater than the threshold will be
        selected. If set to None, all features will be selected. Default is 0.
    checked_label_path : str, optional
        The file path for the corrected label values if the length between features and labels are different. 
        Default is "labels_corrected.csv".

    """
    label: pd.Series | str | Iterable[int|float]
    features: pd.DataFrame | str | list | np.ndarray
    variance_thres: float | None = 0
    checked_label_path: str = "labels_corrected.csv"

    def __post_init__(self):
        self.features = self.read_features(self.features)
        self.labels = self.read_label(self.label)
        self._check_label(self.checked_label_path)
        self.features = self.preprocess()
        self.analyse_composition(self.features)
    
    def _check_label(self, label_path: str | Path) -> None:
        """
        Check that the label data matches the feature data and save it to a file if necessary.

        Parameters
        ----------
        label_path : str or Path
            The path to the label data file.

        Raises
        ------
        KeyError
            If the feature dataframe and labels have different index names.

        Returns
        -------
        None
        """
        if len(self.label) != len(self.features):
            try:
                self.label = self.label.loc[self.features.index]
                label_path = Path(label_path)
                if not label_path.exists():
                    self.label.to_csv(label_path)
            except KeyError as e:
                self.log.error(f"feature dataframe and labels have different index names: {e}")
                raise KeyError(f"feature dataframe and labels have different index names: {e}")
        
    def read_feature(self, features: str | pd.DataFrame | list | np.ndarray) -> pd.DataFrame:
        """
        Read the feature data and return a dataframe.

        Parameters
        ----------
        features : str or pd.DataFrame or list or np.ndarray
            The feature data for the model.

        Raises
        ------
        TypeError
            If the features are not in a valid format.

        Returns
        -------
        pd.DataFrame
            The feature data.
        """
        if isinstance(features, str) and features.endswith(".csv"):
            return pd.read_csv(f"{features}", index_col=0) # the first column should contain the sample names
        elif isinstance(features, pd.DataFrame):
            return features
        elif isinstance(features, (list, np.ndarray)):
            return pd.DataFrame(features)

        self.log.error("features should be a csv file, an array or a pandas DataFrame")
        raise TypeError("features should be a csv file, an array or a pandas DataFrame")

    def read_label(self,  labels: str | pd.Series | Iterable) -> pd.Series:
        """
        Read the label data and return a Series.

        Parameters
        ----------
        labels : pd.Series or str or list or set or np.ndarray
            The label data for the model.

        Raises
        ------
        TypeError
            If the features are not in a valid format.

        Returns
        -------
        pd.Series
            The feature data.
        """
        if isinstance(labels, pd.Series):
            return labels
        elif isinstance(labels, (list, set, np.ndarray)):
            return pd.Series(labels, index=self.features.index, name="target")

        elif isinstance(labels, str) and label.endswith(".csv"):
            if Path(labels).exists():
                return pd.read_csv(labels, index_col=0)
            
            elif labels in self.features.columns:
                label = self.features[labels]
                self.features.drop(labels, axis=1, inplace=True)
                return label
            
        self.log.error("label should be a csv file, a pandas Series, an array or inside features")
        raise TypeError("label should be a csv file, a pandas Series, an array or inside features")

    def preprocess(self) -> pd.DataFrame:
        """
        Eliminate low variance features using the VarianceThreshold from sklearn
        """
        if self.variance_thres is not None:
            variance = VarianceThreshold(self.variance_thres)
            fit = variance.fit_transform(self.features)
            features = pd.DataFrame(fit, index=self.features.index, 
                                         columns=variance.get_feature_names_out())

        return features
    
    def analyse_composition(dataframe):
        col = dataframe.columns
        if col.dtype == int:
            print("num columns: ", len(col))
            return len(col)
        count = col.str.contains("pssm|tpc|eedp|edp").sum()
        if not count:
            print("num columns: ", len(col))
            return len(col)
        
        print(f"POSSUM: {count}, iFeature: {len(col) - count}")
        return count, len(col) - count


class FeatureSelection:
    """
    Class for selecting features from a dataset using different filter methods.
    
    Parameters:

    label : The label values.

    features : The a pandas DataFrame with the feature values.

    excel_file: Path to output Excel file to save selected features. 
    
    num_thread: Number of threads for parallel execution. 
    
    scaler: Name of scaler to use before feature selection.
    
    num_split: Number of folds for KFold cross-validation.
    
    test_size: Test set size for HoldOut validation.  
    
    num_filters: Number of filters to apply for feature selection.
    
    seed: Random seed for reproducibility.

    
    Attributes:
    
    num_thread: Threads for parallelism.
    
    scaler: Chosen scaler.
    
    num_splits: Number of KFold splits. 
    
    test_size: HoldOut test set size.
    
    excel_file: Output Excel file path.
    
    num_filters: Number of filters to apply.
    
    seed: Random seed.
    
    Methods:
    
    feature_set_kfold: KFold feature selection.
    
    feature_set_holdout: HoldOut feature selection.
    
    """
    def __init__(self, features: pd.DataFrame, label: pd.Series, excel_file: str | Path, scaler: str="robust",
                 num_thread: int =10, num_split: int=5, test_size: float=0.2, 
                 num_filters: int=10, seed: int | None=None):
        """
        Initialize a new instance of the FeatureSelection class.

        Parameters
        ----------
        label : pd.Series 
            The label values.
        features : pd.DataFrame
            The a pandas DataFrame with the feature values.
        excel_file : str or Path
            The path to the Excel file to save the selected features.
        features : pd.DataFrame or str or Iterable[list or np.ndarray]
            The input data for the model. Defaults to "training_features/every_features.csv".
        num_thread : int, optional
            The number of threads to use for feature selection. Defaults to 10.
        scaler : str, optional
            The type of scaler to use for feature scaling. Defaults to "robust".
        num_split : int, optional
            The number of splits to use for cross-validation. Defaults to 5.
        test_size : float, optional
            The proportion of data to use for testing. Defaults to 0.2.
        num_filters : int, optional
            The number of filter algorithms to use for feature selection. Defaults to 10.
        seed : int or None, optional
            The random seed to use for reproducibility. Defaults to None.
        """
        # method body
        
        self.log = Log("feature_selection")
        self.log.info("Reading the features")
        self.features = features
        self.label = label
        self.num_thread = num_thread
        self.scaler = scaler
        self.num_splits = num_split
        self.test_size = test_size
        self.excel_file = Path(excel_file)
        self.num_filters = num_filters
        if not str(self.excel_file).endswith(".xlsx"):
            self.excel_file = self.excel_file.with_suffix(".xlsx")
        self.excel_file.parent.mkdir(parents=True, exist_ok=True)
        if seed:
            self.seed = seed
        else:
            self.seed = int(time.time())
        # log parameters
        self.log.info("Starting feature selection and using the following parameters")    
        self.log.info(f"seed: {self.seed}")
        self.log.info(f"Features shape: {self.features.shape}")
        self.log.info(f"Scaler: {self.scaler}")
        self.log.info(f"Variance Threshold: {self.variance_thres}")
        self.log.info(f"Kfold parameters: {self.num_splits}:{self.test_size}")

    def parallel_filter(self, filter_args: dict[str], X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                        num_features: int, feature_names: Iterable[str]) -> pd.DataFrame:
        """
        Perform feature selection using parallelized filters.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            The training feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        num_features : int
            The number of features to select.
        feature_names : list or np.ndarray
            The names of the features.
        filter_args : dict, optional
            A dictionary containing the arguments for each filter. Defaults to an empty dictionary.

        Returns
        -------
        pd.Series
            A series containing the feature scores, sorted in descending order.
        """
        results = {}
        filter_names, multivariate = filter_args["filter_names"], filter_args["multivariate"]
        filter_unsupervised, regression_filters = filter_args["filter_unsupervised"], filter_args["regression_filters"]
        arg_univariate = [(X_train, Y_train, num_features, feature_names, x) for x in filter_names]
        arg_multivariate = [(X_train, Y_train, num_features, feature_names, x) for x in multivariate]
        arg_unsupervised = [(X_train, num_features, feature_names, x) for x in filter_unsupervised]
        arg_regression = [(X_train, Y_train, num_features, feature_names, x) for x in regression_filters]
        
        with get_context("spawn").Pool(self.num_thread) as pool:
            for num, res in enumerate(pool.starmap(methods.univariate, arg_univariate)):
                print(f"univariate filter: {filter_names[num]}")
                results[filter_names[num]] = res

            for num, res in enumerate(pool.starmap(methods.multivariate, arg_multivariate)):
                print(f"multivariate filter: {multivariate[num]}")
                results[multivariate[num]] = res

            for num, res in enumerate(pool.starmap(methods.unsupervised, arg_unsupervised)):
                print(f"unsupervised filter: {filter_unsupervised[num]}")
                results[filter_unsupervised[num]] = res
            
            for num, res in enumerate(pool.starmap(methods.regression_filters, arg_regression)):
                print(f"regression filter: {regression_filters[num]}")
                results[regression_filters[num]] = res
    

        return pd.concat(results)
    
    def supervised_filters(self, filter_args: dict[str], X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                           split_ind: int,  feature_names: Iterable[str], output_path: Path, 
                           plot: bool=True, plot_num_features: int=20) -> pd.DataFrame:
        
        results = {}
        XGBOOST = methods.xgbtree(X_train, Y_train, filter_args["xgboost"], self.seed, self.num_thread)
        shap_importance, shap_values = methods.calculate_shap_importance(XGBOOST, X_train, Y_train, feature_names)
        if plot:
            methods.plot_shap_importance(shap_values, feature_names, output_path, split_ind, X_train, plot_num_features)
        results["xgbtree"] = shap_importance
        results["random"] = methods.random_forest(X_train, Y_train, feature_names, filter_args["random_forest"], 
                                                  self.seed, self.num_thread)

        return pd.concat(results)

    def _write_dict(self, feature_dict: dict) -> None:

        """
        Write the constructed feature sets to an Excel file.

        Parameters
        ----------
        feature_dict : dict
            A dictionary containing the constructed feature sets.

        Returns
        -------
        None
        """
        # TODO: Maybe change it to list(value.values())[0] so It is not a multiindex column for holdout
        final_dict = {key: pd.concat(value, axis=1) for key, value in feature_dict.items()}
        with pd.ExcelWriter(self.excel_file, mode="w", engine="openpyxl") as writer:
            for key in final_dict.keys():
                write_excel(writer, final_dict[key], key)

    def generate_features(self, filter_args: dict, X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                          feature_range: Iterable[int], feature_dict: dict, split_ind: int, rfe_step: int=30, plot: bool=True,
                          plot_num_features: int=20) -> None:
        """
        Construct a dictionary of feature sets using univariate filters and recursive feature elimination.

        Parameters
        ----------
        univariate_features : pd.DataFrame
            A dataframe containing the univariate filter scores for each feature.
        features : pd.DataFrame
            The original feature data.
        feature_dict : dict
            A dictionary to store the constructed feature sets.
        feature_range : list or np.ndarray
            A range of the number of features to include in each feature set.
        transformed : pd.DataFrame
            The transformed feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        split_ind : int
            The index of the current split.
        rfe_step : int, optional
            The number of features to remove at each iteration in recursive feature elimination. Defaults to 30.

        Returns
        -------
        None
        """

        transformed, scaler_dict = scale(self.scaler, X_train)
        # for each split I do again feature selection and save all the features selected from different splits
        # but with the same selector in the same dictionary
        self.log.info("filtering the features")
        univariate_features = self.parallel_filter(filter_args, transformed, Y_train, feature_range[-1],
                                                    self.features.columns)
        supervised_features = self.supervised_filters(filter_args, transformed, Y_train, split_ind, self.features.columns, 
                                                        self.excel_file.parent, plot, plot_num_features)
        
        concatenated_features = pd.concat([univariate_features, supervised_features])
        for num_features in feature_range:
            print(f"generating a feature set of {num_features} dimensions")
            for filters in concatenated_features.index.unique(0):
                feat = concatenated_features.loc[filters]
                feature_dict[f"{filters}_{num_features}"][f"split_{split_ind}"] = self.features[feat.index[:num_features]]
            rfe_results = methods.rfe_linear(transformed, Y_train, num_features, self.features.columns, rfe_step, 
                                             filter_args["RFE"])
            feature_dict[f"rfe_{num_features}"][f"split_{split_ind}"] = self.features[rfe_results]

        return feature_dict


    def feature_set_kfold(self, filter_args: dict, split_function: ShuffleSplit | StratifiedShuffleSplit, 
                          feature_range, rfe_step: int=30, plot: bool=True, 
                          plot_num_features: int=20) -> None:
        """
        Perform feature selection using k-fold cross-validation.

        Parameters
        ----------
        filter_args : dict
            A dictionary containing the arguments for each filter algorithm.
        feature_range : list[int]
            A range of the number of features to include in each feature set.
        rfe_step : int, optional
            The number of features to remove at each iteration in recursive feature elimination. Defaults to 30.
        plot : bool, optional
            Whether to plot the feature importances. Defaults to True.
        plot_num_features : int, optional
            The number of features to include in the plot. Defaults to 20.

        Returns
        -------
        featre_dict : dict[str, pd.DataFrame]
        """
        feature_dict = defaultdict(dict)
        for split_index, (train_index, test_index) in enumerate(split_function.split(self.features, self.label)):
            self.log.info(f"kfold {split_index}")
            self.log.info("------------------------------------")
            X_train = self.features.iloc[train_index]
            Y_train = self.label.iloc[train_index].values.ravel()
            self.generate_features(filter_args, X_train, Y_train, feature_range, feature_dict, split_index, rfe_step, plot,
                                   plot_num_features)

        return feature_dict

class FeatureClassification(FeatureSelection):
    def __init__(self, label, excel_file, features, num_thread=10, scaler="robust", 
                 num_split=5, test_size=0.2, num_filters=10, seed=None):
        
        super().__init__(label, features, excel_file, num_thread, scaler, num_split, test_size, num_filters, seed)
        """Subclass to perform feature selection on classification problems with a set of predefined filter methods"""
        random.seed(self.seed)
        filter_names = ("FRatio", "SymmetricUncertainty", "SpearmanCorr", "PearsonCorr", "Chi2", "Anova",
                        "LaplacianScore", "InformationGain", "KendallCorr", "FechnerCorr")
        filter_names = random.sample(filter_names, self.num_filters)
        multivariate = ("STIR", "TraceRatioFisher")
        filter_unsupervised = ("TraceRatioLaplacian",)
        self._filter_args = {"filter_names": filter_names, "multivariate": multivariate, "xgboost": xgb.XGBClassifier, 
                             "RFE": RidgeClassifier, "random_forest": rfc,
                            "filter_unsupervised": filter_unsupervised, "regression_filters":()}

        self.log.info("Classification Problem")
        self.log.info(f"Using {len(filter_names)+len(multivariate)+len(filter_unsupervised)} 
                      filters: {filter_names}, {filter_unsupervised} and {multivariate}")

    @property
    def filter_args(self):
        return self._filter_args
    
    @filter_args.setter
    def filter_args(self, value: tuple[str, Iterable[str]] | dict[str, Iterable[str]]):
        if isinstance(value, dict):
            dif = set(value.keys()).difference(self._filter_args.keys())
            if dif:
                raise KeyError(f"these keys: {dif} are not valid")
            self._filter_args = value
        elif isinstance(value, tuple):
            self._filter_args[value[0]] = tuple(value[1])

    def construct_kfold(self, feature_range, rfe_step=30, plot=True, plot_num_features=20):

        skf = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.seed)
        feature_dict = self.feature_set_kfold(self.filter_args, skf, feature_range, rfe_step, plot, plot_num_features)
        self._write_dict(feature_dict)

    def construct_holdout(self, feature_range, plot=True, plot_num_features=20, rfe_step=30):
        """
        Perform feature selection using a holdout strategy.

        Parameters
        ----------
        filter_args : dict
            A dictionary containing the arguments for each filter.
        feature_range : Iterable[int]
            A range of feature to select
        plot : bool, optional
            Whether to plot the feature importances. Defaults to True.
        plot_num_features : int, optional
            The number of features to include in the plot. Defaults to 20.
        rfe_step : int, optional
            The number of features to remove at each iteration in recursive feature elimination. Defaults to 30.

        Returns
        -------
        None
        """
        
        feature_dict = defaultdict(dict)
        X_train, X_test, Y_train, Y_test = train_test_split(self.features, self.label, test_size=self.test_size, 
                                                                random_state=self.seed, stratify=self.label)
        self.generate_features(self.filter_args, X_train, Y_train, feature_range, feature_dict, 0, rfe_step, plot,
                               plot_num_features)
        self._write_dict(feature_dict)


class FeatureRegression(FeatureSelection):
    """Subclass to perform feature selection on regression problems with a set of predefined filter methods"""

    def __init__(self, label, excel_file, features, num_thread=10, scaler="robust", num_split=5, 
                 test_size=0.2, num_filters=4, seed=None):
        
        super().__init__(label, features, excel_file, num_thread, scaler, num_split, test_size, 
                         num_filters, seed)
        
        random.seed(self.seed)
        filter_names = ("SpearmanCorr", "PearsonCorr", "KendallCorr", "FechnerCorr")
        filter_names = random.sample(filter_names, self.num_filters)
        regression_filters = ("mutual_info", "Fscore")
        self._filter_args = {"filter_names": filter_names, "multivariate": (), "xgboost": xgb.XGBRegressor, 
                             "RFE": Ridge, "random_forest": rfr,
                            "filter_unsupervised": (), "regression_filters": regression_filters}
        self.log.info("Regression Problem")
        self.log.info(f"Using {len(filter_names) + len(regression_filters)} filters: {filter_names} and {regression_filters}")

    @property
    def filter_args(self):
        return self._filter_args
    
    @filter_args.setter
    def filter_args(self, value: tuple[str, Iterable[str] | Ridge | RidgeClassifier | xgb.XGBRegressor | xgb.XGBClassifier]):
        self._filter_args[value[0]] = value[1]

    def construct_kfold(self, feature_range, rfe_step=30, plot=True, plot_num_features=20):
         
         skf = ShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.seed)
         feature_dict = self.feature_set_kfold(self.filter_args, skf, feature_range, rfe_step, plot, plot_num_features)
         self._write_dict(feature_dict)

    def construct_holdout(self, feature_range: Iterable[int], plot=True, plot_num_features=20, rfe_step=30):
        """
        Perform feature selection using a holdout strategy.

        Parameters
        ----------
        filter_args : dict
            A dictionary containing the arguments for each filter.
        feature_range : Iterable[int]
            A range of feature to select
        plot : bool, optional
            Whether to plot the feature importances. Defaults to True.
        plot_num_features : int, optional
            The number of features to include in the plot. Defaults to 20.
        rfe_step : int, optional
            The number of features to remove at each iteration in recursive feature elimination. Defaults to 30.

        Returns
        -------
        None
        """

        feature_dict = defaultdict(dict)
        X_train, X_test, Y_train, Y_test = train_test_split(self.features, self.label, test_size=self.test_size, 
                                                            random_state=self.seed)
        self.generate_features(self.filter_args, X_train, Y_train, feature_range, feature_dict, 0, rfe_step, plot,
                               plot_num_features)
        self._write_dict(feature_dict)


def get_range_features(self, num_features_min: int | None=None, 
                        num_features_max: int | None=None, step_range: int | None=None) -> list:
    """
    Get a range of features dimensions to select.

    Parameters
    ----------
    num_features_min : int, optional
        The minimum number of features to select. If not provided, defaults to 1/10 of the total number of features.
    num_features_max : int, optional
        The maximum number of features to select. If not provided, defaults to 1/2 of the total number of features + 1.
    step_range : int, optional
        The step size for the range of numbers. If not provided, defaults to 1/4 of the difference between the minimum
        and maximum number of features.

    Returns
    -------
    list
        A list of integers representing the range of numbers for the number of features to select.
    """
    if not num_features_min:
        num_features_min = len(self.features.columns) // 10
        if not num_features_max:
            num_features_max = len(self.features.columns) // 2 + 1
        if not step_range:
            step_range = (num_features_max - num_features_min) // 4
        feature_range = list(range(num_features_min, num_features_max, step_range))
    elif num_features_min and step_range and num_features_max:
        feature_range = list(range(num_features_min, num_features_max, step_range))
    else:
        feature_range = [num_features_min]

    return feature_range


def translate_range_str_to_list(feature_range):
    # translate the argments passsed as strings to a list of integers
    feature_range = feature_range.split(":")
    num_features_min, num_features_max, step = feature_range
    if num_features_min.lower() == "none":
        num_features_min = None
        steps = None
        nums_features_max = None
    else:
        num_features_min = int(num_features_min)
        steps = None
        if step.isdigt():
            steps = int(step)

        nums_features_max = None
        if num_features_max.isdigit():
            nums_features_max = int(num_features_max)

    return num_features_min, nums_features_max, steps


def main():
    features, label, variance_threshold, feature_range, num_thread, scaler, excel_file, kfold, rfe_steps, plot, \
        plot_num_features, num_filters, seed, strategy, problem = arg_parse()
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])

    num_features_min, num_features_max, step = translate_range_str_to_list(feature_range)
    feature_range = get_range_features(num_features_min, num_features_max, step)

    # generate the dataset
    training_data = DataReader(label, features, variance_threshold)
    feature = training_data.features
    labels = training_data.labels

    # select features
    if problem == "classification":
        selection = FeatureClassification(labels, excel_file, feature, num_thread, scaler,
                                          num_split, test_size, num_filters, seed)
    elif problem == "regression":
        selection = FeatureRegression(labels, excel_file, feature, num_thread, scaler,
                                      num_split, test_size, num_filters, seed)

    if strategy == "holdout":
        selection.construct_holdout(feature_range, rfe_steps, plot, plot_num_features)
    elif strategy == "kfold":
        selection.construct_kfold(feature_range, rfe_steps, plot, plot_num_features)

   

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()