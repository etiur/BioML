import argparse
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from dataclasses import dataclass
from pathlib import Path
from sklearn.linear_model import RidgeClassifier, Ridge
import numpy as np
from multiprocessing import get_context  # https://pythonspeed.com/articles/python-multiprocessing/
import time
from sklearn.model_selection import train_test_split
from typing import Iterable, Any
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
import xgboost as xgb
from ..utilities.utils import Log, scale, write_excel
from ..utilities.custom_errors import DifferentLabelFeatureIndexError
from . import methods


def arg_parse():
    parser = argparse.ArgumentParser(description="Preprocess and Select the best features, only use it if the feature came from possum or ifeatures")

    parser.add_argument("-f", "--features", required=True,
                        help="The path to the training features")
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
    parser.add_argument("-ts", "--test_size", required=False, type=float,
                        help="The test size of the split", default=0.2)
    parser.add_argument("-rt", "--rfe_steps", required=False, type=int,
                        help="The number of steps for the RFE algorithm, the more step the more precise "
                             "but also more time consuming and might lead to overfitting", default=30)
    parser.add_argument("-p", "--plot", required=False, action="store_false",
                        help="Default to true, plot the feature importance using shap")
    parser.add_argument("-pk", "--plot_num_features", required=False, default=20, type=int,
                        help="How many features to include in the plot")
    parser.add_argument("-se", "--seed", required=False, type=int, default=None,
                        help="The seed number used for reproducibility")
    parser.add_argument("-pr", "--problem", required=True, choices=("classification", "regression"), 
                        default="classification", help="Classification or Regression problem")

    args = parser.parse_args()

    return [args.features, args.label, args.variance_threshold, args.feature_range, args.num_thread, args.scaler,
            args.excel_file, args.test_size, args.rfe_steps, args.plot, args.plot_num_features,
            args.seed, args.problem]

@dataclass(slots=True)
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
    sheet: str or int or None, optional
        The sheet_name for the excel file. Default is None.

    Attributes
    ----------
    label : pd.Series
        The label column of the CSV file.
    features : pd.DataFrame
        The feature columns of the CSV file.
    variance_thres : float or None
        The variance threshold for feature selection.
    checked_label_path : str
        The file path for the corrected label values if the length between features and labels are different.
    sheet: str or int
        The sheet_name for the excel file

    Methods
    -------
    preprocess()
        Preprocesses the feature data using the specified scaler.
    read_features(features)
        Reads the feature data from the specified source.
    read_label(label)
        Reads the label data from the specified source.
    _check_label(checked_label_path)
        Checks if the length of the label data matches the length of the feature data.
    analyse_composition(features)
        Analyses the composition of the feature data.
    """
    label: pd.Series | pd.DataFrame | str | Iterable[int|float]
    features: pd.DataFrame | str | list | Iterable[int|float]
    variance_thres: float | None = 0
    checked_label_path: str = "labels_corrected.csv"
    sheet: str | int | None = None

    def __post_init__(self):

        self.features = self.read_multiple_features(self.features)
        self.label = self.read_label(self.label)
        self._check_label(self.checked_label_path)
        self.label = self.label.to_numpy().flatten()
        self.preprocess()
    
    def _check_label(self, label_path: str | Path) -> None:
        """
        Check that the label data matches the feature data and save it to a file if necessary.

        Parameters
        ----------
        label_path : str or Path
            The path to the label data file.

        Raises
        ------
        DifferentLabelFeatureIndexError
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
                raise DifferentLabelFeatureIndexError("feature dataframe and labels have different index names") from e

    def read_multiple_features(self, features) -> pd.DataFrame:
        """
        Read multiple feature files and concatenate them into a single dataframe.

        Parameters
        ----------
        features : list[str]
            A list of file paths to the feature data or any other formats.

        Returns
        -------
        pd.DataFrame
            _description_
        """
        match features:
            case [*feat] if isinstance(feat[0], (int, float)):
                featu = self.read_feature(feat)
                return featu
            case list(feat):
                data = [self.read_feature(x) for x in feat]
                index = data[0].index
                for num, d in enumerate(data[1:], 1):
                    if not d.index.equals(index):
                        print("features don't have the same index, trying to make them equal")
                        d.index = index
                        data[num] = d
                featu = pd.concat(data, axis=1)
                return featu
            case _:
                featu = self.read_feature(self.features)
                return featu

    def read_feature(self, features: str | pd.DataFrame | list | np.ndarray) -> pd.DataFrame:
        """
        Read the feature data and return a dataframe.

        Parameters
        ----------
        features : str or pd.DataFrame or list or np.ndarray
            The feature data for the model.

        Raises
        ------
        ValuError
            If the features are not in a valid format.

        Returns
        -------
        pd.DataFrame
            The feature data.
        """
        match features:
            case str(feat) if feat.endswith(".csv"):
                return pd.read_csv(f"{features}", index_col=0) # the first column should contain the sample names
            case str(feat) if feat.endswith(".xlsx"):
                sheet = self.sheet if self.sheet else 0
                return pd.read_excel(f"{features}", index_col=0, sheet_name=sheet) # the first column should contain the sample names
            case pd.DataFrame() as feat:
                return feat
            case list() | np.ndarray() | dict() as feat:
                return pd.DataFrame(feat)
            case _:
                raise ValueError("features should be a csv file, an array or a pandas DataFrame")

    def read_label(self,  labels: str | pd.Series | Iterable[int]) -> pd.Series | pd.DataFrame:
        """
        Read the label data and return a Series.

        Parameters
        ----------
        labels : pd.Series or str or list or set or np.ndarray
            The label data for the model.

        Raises
        ------
        NotSupportedDataError
            If the features are not in a valid format.

        Returns
        -------
        pd.Series
            The feature data.
        """
        match labels: 
            case pd.Series() | pd.DataFrame() as label:
                return label
            case str(label) if label.endswith(".csv"): 
                return pd.read_csv(f"{label}", index_col=0) # the first column should contain the sample names
            case str(label) if label in self.features.columns:            
                lab = self.features[label]
                self.features.drop(label, axis=1, inplace=True)
                return lab
            case list() | np.ndarray() as label:
                return pd.Series(label, index=self.features.index, name="target")
            case _:
                raise ValueError("label should be a csv file, a pandas Series, DataFrame, an array or inside features")

    def preprocess(self) -> None:
        """
        Eliminate low variance features using the VarianceThreshold from sklearn
        """
        if self.variance_thres is not None:
            variance = VarianceThreshold(self.variance_thres)
            fit = variance.fit_transform(self.features)
            self.features = pd.DataFrame(fit, index=self.features.index, 
                                         columns=variance.get_feature_names_out())
    
    def __repr__(self):
        string = f"""Data with:\n    num. samples: {len(self.features)}\n    num. columns: {len(self.features.columns)}\n    variance threshold: {self.variance_thres}\n    sheet: {self.sheet}"""
        return string


class FeatureSelection:
    def __init__(self, excel_file: str | Path, num_thread: int =10, seed: int | None=None):
        """
        A class for performing feature selection on a dataset.

        Parameters
        ----------
        excel_file : str or Path
            The path to the Excel file to save the selected features.
        num_thread : int, optional
            The number of threads to use for feature selection. Defaults to 10.
        seed : int or None, optional
            The random seed to use for reproducibility. Defaults to None.
        """
        # method body
        
        self.log = Log("feature_selection")
        self.log.info("Reading the features")
        self.num_thread = num_thread
        self.excel_file = Path(excel_file)
        if not str(self.excel_file).endswith(".xlsx"):
            self.excel_file = self.excel_file.with_suffix(".xlsx")
        self.excel_file.parent.mkdir(parents=True, exist_ok=True)
        if seed: self.seed = seed
        else: self.seed = int(time.time())
        # log parameters
        self.log.info("Starting feature selection and using the following parameters")    
        self.log.info(f"seed: {self.seed}")

    def parallel_filter(self, filter_args: dict[str, Any], X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                        num_features: int, feature_names: Iterable[str]) -> pd.DataFrame:
        """
        Perform parallelized feature selection.

        Parameters
        ----------
        filter_args : dict[str]
            A dictionary containing the arguments for the filter method.
        X_train : pd.DataFrame or np.ndarray
            The training feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        num_features : int
            The number of features to select.
        feature_names : Iterable[str]
            The names of the features.

        Returns
        -------
        pd.DataFrame
            The selected features.
        """
        results = {}
        filter_names, regression_filters = filter_args["filter_names"], filter_args["regression_filters"]
        arg_univariate = [(X_train, Y_train, num_features, feature_names, x) for x in filter_names]
        arg_regression = [(X_train, Y_train, num_features, feature_names, x) for x in regression_filters]
        
        with get_context("spawn").Pool(self.num_thread) as pool:
            for num, res in enumerate(pool.starmap(methods.classification_filters, arg_univariate)):
                print(f"classification filter: {filter_names[num]}")
                results[filter_names[num]] = res
            
            for num, res in enumerate(pool.starmap(methods.regression_filters, arg_regression)):
                print(f"regression filter: {regression_filters[num]}")
                results[regression_filters[num]] = res
    

        return pd.concat(results)
    
    def supervised_filters(self, filter_args: dict[str, Any], X_train: pd.DataFrame | np.ndarray, 
                           Y_train: pd.Series | np.ndarray, 
                           feature_names: Iterable[str], output_path: Path, plot: bool=True, 
                           plot_num_features: int=20) -> pd.DataFrame:
        """
        Perform feature selection using supervised filter methods.

        Parameters
        ----------
        filter_args : dict[str]
            A dictionary containing the arguments for the filter method.
        X_train : pd.DataFrame or np.ndarray
            The training feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        feature_names : Iterable[str]
            The names of the features.
        output_path : Path
            The path to save the output plots.
        plot : bool, optional
            Whether to plot the feature importances. Defaults to True.
        plot_num_features : int, optional
            The number of features to plot. Defaults to 20.

        Returns
        -------
        pd.DataFrame
            The selected features.
        """
        results = {}
        shap_importance, shap_values = methods.xgbtree(X_train, Y_train, filter_args["xgboost"], feature_names,
                                                       self.seed, self.num_thread)
        if plot:
            methods.plot_shap_importance(shap_values, feature_names, output_path, X_train, plot_num_features)
        results["xgbtree"] = shap_importance
        results["random"] = methods.random_forest(X_train, Y_train, feature_names, filter_args["random_forest"], 
                                                  self.seed, self.num_thread)

        return pd.concat(results)

    def _write_dict(self, feature_dict: dict[str, pd.DataFrame]) -> None:

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

        with pd.ExcelWriter(self.excel_file, mode="w", engine="openpyxl") as writer:
            for key, value in feature_dict.items():
                write_excel(writer, value, key)

    def generate_features(self, filter_args: dict[str, Any], transformed: np.ndarray, Y_train: pd.Series | np.ndarray, 
                          feature_range: list[int], features: pd.DataFrame, rfe_step: int=30, plot: bool=True,
                          plot_num_features: int=20) -> dict[str, pd.DataFrame]:
        """
        Construct a dictionary of feature sets using all features.

        Parameters
        ----------
        filter_args : dict[str]
            A dictionary containing the arguments for the filter method.
        transformed : np.ndarray
            The transformed feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        feature_range : Iterable[int]
            A range of the number of features to include in each feature set.
        features : pd.DataFrame
            The original feature data.
        rfe_step : int, optional
            The number of features to remove at each iteration in recursive feature elimination. Defaults to 30.
        plot : bool, optional
            Whether to plot the feature importances. Defaults to True.
        plot_num_features : int, optional
            The number of features to plot. Defaults to 20.

        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary containing the constructed feature sets.
        """
        feature_dict = {}

        self.log.info("filtering the features")
        univariate_features = self.parallel_filter(filter_args, transformed, Y_train, feature_range[-1],
                                                   features.columns)
        supervised_features = self.supervised_filters(filter_args, transformed, Y_train, features.columns, 
                                                        self.excel_file.parent, plot, plot_num_features)
        
        concatenated_features = pd.concat([univariate_features, supervised_features])
        for num_features in feature_range:
            print(f"generating a feature set of {num_features} dimensions")
            for filters in concatenated_features.index.unique(0):
                feat = concatenated_features.loc[filters]
                feature_dict[f"{filters}_{num_features}"]= features[feat.index[:num_features]]
            rfe_results = methods.rfe_linear(transformed, Y_train, num_features, self.seed, features.columns, rfe_step, 
                                             filter_args["RFE"])
            feature_dict[f"rfe_{num_features}"] = features[rfe_results]

        return feature_dict


class FeatureClassification:
    """
    A class for performing feature selection for classification problems.

    Parameters
    ----------
    seed : int
        The random seed to use for reproducibility.
    scaler : str, optional
        The type of scaler to use for preprocessing the features. Can be "standard",
        "minmax", or "robust". Default is "robust".
    test_size : float, optional
        The proportion of data to use for testing. Defaults to 0.2.

    Attributes
    ----------
    log : Log
        The logger for the ActiveSelection class.
    seed : int
        The random seed to use for reproducibility.
    num_filters : int
        The number of filter algorithms to use for feature selection.
    test_size : float
        The proportion of data to use for testing.

    Methods
    -------
    construct_features(features, selector, feature_range, plot=True, plot_num_features=20, rfe_step=30)
        Perform feature selection using a holdout strategy.
    """
    def __init__(self, seed: int, scaler: str = "robust", test_size: float = 0.2):
        
        """
        Class to perform feature selection on classification problems with a set of predefined filter methods
        """

        self.seed = seed
        self.scaler = scaler

        filter_names = ("mutual_info", "Fscore", "chi2", "FechnerCorr", "KendallCorr")
        self._filter_args = {"filter_names": filter_names, "xgboost": xgb.XGBClassifier, 
                             "RFE": RidgeClassifier, "random_forest": rfc,
                             "regression_filters":()}
        self.test_size = test_size

    @property
    def filter_args(self):
        """
        Get the dictionary of filter arguments.

        Returns
        -------
        dict[str, Iterable[str]]
            A dictionary containing the arguments for the filter methods.
        """
        return self._filter_args
    
    @filter_args.setter
    def filter_args(self, value: tuple[str, Iterable[str]] | dict[str, Iterable[str]]):
        """
        Set the dictionary of filter arguments.

        Parameters
        ----------
        value : tuple[str, Iterable[str]] or dict[str, Iterable[str]]
            A dictionary containing the arguments for the filter methods, or a tuple containing the filter name and its arguments.

        Raises
        ------
        KeyError
            If the filter name is not a valid filter.
        """
        match value:
            case dict(filter_args):
                for key, val in filter_args.items():
                    if key not in self._filter_args:
                        raise KeyError(f"filter {key} is not a valid filter")
                    self._filter_args[key] = val
            case tuple(filter_args):
                self._filter_args[filter_args[0]] = tuple(filter_args[1])

    def construct_features(self, features: pd.DataFrame | np.ndarray, label: Iterable[int], 
                           selector: FeatureSelection, feature_range: list[int], 
                           plot: bool=True, plot_num_features: int=20, rfe_step: int=30):
        """
        Perform feature selection using a holdout strategy.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            The training features
        label : Iterable[int]
            The training labels
        selector : FeatureSelection
            The feature selection object to use.
        feature_range : list[int]
            A list of the number of features to select at each iteration.
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
        
        X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size=self.test_size, 
                                                            random_state=self.seed, stratify=label)
        
        transformed_x, scaler_dict = scale(self.scaler, X_train)
        feature_dict = selector.generate_features(self.filter_args, transformed_x, Y_train, 
                                                  feature_range, features, rfe_step, plot,
                                                  plot_num_features)
        selector._write_dict(feature_dict)


class FeatureRegression:
    """
    A Class to perform feature selection on regression problems with a set of predefined filter methods.

    Parameters
    ----------
    seed : int
        The random seed to use for reproducibility.
    scaler : str, optional
        The type of scaler to use for preprocessing the features. Can be "standard",
        "minmax", or "robust". Default is "robust".
    test_size : float, optional
        The proportion of data to use for testing. Defaults to 0.2.

    Attributes
    ----------
    log : Log
        The logger for the ActiveSelection class.
    seed : int
        The random seed to use for reproducibility.
    num_filters : int
        The number of filter algorithms to use for feature selection.
    test_size : float
        The proportion of data to use for testing.

    Methods
    -------
    construct_features(features, selector, feature_range, plot=True, plot_num_features=20, rfe_step=30)
        Perform feature selection using a holdout strategy.
    """
    def __init__(self, seed: int, scaler: str="robust", test_size=0.2):
        self.seed = seed
        self.scaler = scaler
        regression_filters = ("mutual_info", "Fscore", "PCC")
        self._filter_args = {"filter_names": (), "xgboost": xgb.XGBRegressor, 
                             "RFE": Ridge, "random_forest": rfr,
                             "regression_filters": regression_filters}
        
        self.test_size = test_size

    @property
    def filter_args(self):
        return self._filter_args
    
    @filter_args.setter
    def filter_args(self, value: tuple[str, Iterable[str]] | dict[str, Iterable[str]]):
        match value:
            case dict(filter_args):
                for key, val in filter_args.items():
                    if key not in self._filter_args:
                        raise KeyError(f"filter {key} is not a valid filter")
                    self._filter_args[key] = val
            case tuple(filter_args):
                self._filter_args[filter_args[0]] = tuple(filter_args[1])

    def construct_features(self, features: pd.DataFrame | np.ndarray, label: Iterable[int|float],  selector: FeatureSelection, 
                           feature_range: list[int], plot: bool=True, plot_num_features:int=20, 
                           rfe_step:int=30) -> None:
        """
        Perform feature selection using a holdout strategy.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            The feature data.
        label : Iterable[int|float]
            The regression label data.
        selector : FeatureSelection
            The feature selection object to use.
        feature_range : list[int]
            A list of the number of features to select at each iteration.
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

        X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size=self.test_size, 
                                                            random_state=self.seed)
        transformed_x, scaler_dict = scale(self.scaler, X_train)
        feature_dict = selector.generate_features(self.filter_args, transformed_x, Y_train, 
                                                  feature_range, features, rfe_step, 
                                                  plot, plot_num_features)
        selector._write_dict(feature_dict)

   
def get_range_features(features: pd.DataFrame, num_features_min: int | None=None, 
                        num_features_max: int | None=None, step_range: int | None=None) -> list[int]:
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
        if not num_features_max:
            num_features_min = max(2, len(features.columns) // 10)
            num_features_max = max(4, int(len(features.columns) // 1.6) + 1)
        else:
            num_features_min = max(2, num_features_max // 5)
            num_features_max = max(4, num_features_max)
        if not step_range:
            step_range = max(1, (num_features_max - num_features_min) // 4)
        feature_range = list(range(num_features_min, num_features_max, step_range))
        return feature_range + [num_features_max]
    elif num_features_min and step_range and num_features_max:
        feature_range = list(range(num_features_min, num_features_max, step_range))
        return feature_range
    else:
        feature_range = [num_features_min]
        return feature_range


def translate_range_str_to_list(feature_range: str) -> tuple[int | None, int | None, int | None]:
    """
    Translates the arguments passed as strings to a list of integers.

    Parameters
    ----------
    feature_range : str
        The feature range string in the format "num_features_min:num_features_max:step".

    Returns
    -------
    tuple[int | None, int | None, int | None]
        A tuple containing the minimum number of features, the maximum number of features, and the step size.
    """
    # translate the argments passsed as strings to a list of integers
    num_features_min, num_features_max, step = feature_range.split(":")
    if num_features_min.lower() == "none":
        num_features_min = None
        steps = None
        nums_features_max = None
    else:
        num_features_min = int(num_features_min)
        steps = None
        if step.isdigit():
            steps = int(step)

        nums_features_max = None
        if num_features_max.isdigit():
            nums_features_max = int(num_features_max)

    return num_features_min, nums_features_max, steps


def main():
    features, label, variance_threshold, feature_range, num_thread, scaler, excel_file, test_size, rfe_steps, plot, \
        plot_num_features, seed, problem = arg_parse()

    num_features_min, num_features_max, step = translate_range_str_to_list(feature_range)

    # generate the dataset
    training_data = DataReader(label, features, variance_threshold)
    feature_range = get_range_features(training_data.features, num_features_min, num_features_max, step)
    filters = FeatureSelection(excel_file, num_thread, seed)
    # select features
    if problem == "classification":
        selection = FeatureClassification(filters.seed, scaler, test_size)
    elif problem == "regression":
        selection = FeatureRegression(filters.seed, scaler, test_size)

    selection.construct_features(training_data.features, training_data.label, filters, feature_range, 
                                 rfe_steps, plot, plot_num_features)

   

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()