import argparse
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from ..utilities import Log, scale, write_excel
from pathlib import Path
# Multiprocess instead of Multiprocessing solves the pickle problem in Windows (might be different in linux)
# but it has its own errors. Use multiprocessing.get_context('fork') seems to solve the problem but only available
# in Unix. Altough now it seems that with fork the programme can hang indefinitely so use spaw instead
# https://medium.com/devopss-hole/python-multiprocessing-pickle-issue-e2d35ccf96a9
import xgboost as xgb
from sklearn.linear_model import RidgeClassifier, Ridge
import numpy as np
import random
from multiprocessing import get_context  # https://pythonspeed.com/articles/python-multiprocessing/
import time
from sklearn.model_selection import train_test_split
from typing import Iterable, Any
from dataclasses import dataclass
from . import methods
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from ..custom_errors import NotSupportedError


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
                        help="The number univariate filters to use maximum 10 for classification and 4 for regression", default=10)
    parser.add_argument("-se", "--seed", required=False, type=int, default=None,
                        help="The seed number used for reproducibility")
    parser.add_argument("-st", "--strategy", required=False, choices=("holdout", "kfold"), default="holdout",
                        help="The spliting strategy to use")
    parser.add_argument("-pr", "--problem", required=True, choices=("classification", "regression"), 
                        default="classification", help="Classification or Regression problem")

    args = parser.parse_args()

    return [args.features, args.label, args.variance_threshold, args.feature_range, args.num_thread, args.scaler,
            args.excel_file, args.kfold_parameters, args.rfe_steps, args.plot, args.plot_num_features, args.num_filters,
            args.seed, args.strategy, args.problem]

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
    scaler : str, optional
        The type of scaler to use for preprocessing the features. Can be "standard",
        "minmax", or "robust". Default is "robust".
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
    scaler : str
        The type of scaler used for preprocessing the features.
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
    features: pd.DataFrame | str | list | np.ndarray
    variance_thres: float | None = 0
    scaler: str = "robust"
    checked_label_path: str = "labels_corrected.csv"
    sheet: str | int | None = None

    def __post_init__(self):
        self.features = self.read_feature(self.features)
        self.label = self.read_label(self.label)
        self._check_label(self.checked_label_path)
        self.label = self.label.to_numpy().flatten()
        self.preprocess()
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
        NotSupportedError
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
            case list() | np.ndarray() as feat:
                return pd.DataFrame(feat)
            case _:
                raise NotSupportedError("features should be a csv file, an array or a pandas DataFrame")

    def read_label(self,  labels: str | pd.Series | Iterable[int]) -> pd.Series | pd.DataFrame:
        """
        Read the label data and return a Series.

        Parameters
        ----------
        labels : pd.Series or str or list or set or np.ndarray
            The label data for the model.

        Raises
        ------
        NotSupportedError
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
                raise NotSupportedError("label should be a csv file, a pandas Series, DataFrame, an array or inside features")

    def preprocess(self) -> None:
        """
        Eliminate low variance features using the VarianceThreshold from sklearn
        """
        if self.variance_thres is not None:
            variance = VarianceThreshold(self.variance_thres)
            fit = variance.fit_transform(self.features)
            self.features = pd.DataFrame(fit, index=self.features.index, 
                                         columns=variance.get_feature_names_out())


    

    def analyse_composition(self, dataframe: pd.DataFrame) -> int | tuple[int, int]:
        """
        Analyses the composition of the given pandas DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to analyse.

        Returns
        -------
        Union[int, Tuple[int, int]]
            If all columns are numeric, returns the number of columns.
            If there are non-numeric columns, returns a tuple containing the number
            of columns containing "pssm", "tpc", "eedp", or "edp" and the number of
            columns not containing those substrings.
        """
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
    
    def scale(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Scales the training data using the specified scaler.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data to scale.

        Returns
        -------
        np.ndarray
            The scaled training data as a numpy array.
        """
        transformed, scaler_dict = scale(self.scaler, X_train)
        return transformed.to_numpy()


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
        if seed:
            self.seed = seed
        else:
            self.seed = int(time.time())
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
        XGBOOST = methods.xgbtree(X_train, Y_train, filter_args["xgboost"], self.seed, self.num_thread)
        shap_importance, shap_values = methods.calculate_shap_importance(XGBOOST, X_train, Y_train, feature_names)
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
    num_filters : int, optional
        The number of filter algorithms to use for feature selection. Defaults to 10.
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
    def __init__(self, seed: int, num_filters: int=10, test_size = 0.2):
        
        """Class to perform feature selection on classification problems with a set of predefined filter methods"""
        self.seed = seed
        random.seed(self.seed)
        filter_names = ("FRatio", "SymmetricUncertainty", "SpearmanCorr", "PearsonCorr", "Chi2", "Anova",
                        "LaplacianScore", "InformationGain", "KendallCorr", "FechnerCorr")
        filter_names = random.sample(filter_names, num_filters)
        multivariate = ("STIR", "TraceRatioFisher")
        filter_unsupervised = ("TraceRatioLaplacian",)
        self._filter_args = {"filter_names": filter_names, "multivariate": multivariate, "xgboost": xgb.XGBClassifier, 
                             "RFE": RidgeClassifier, "random_forest": rfc,
                            "filter_unsupervised": filter_unsupervised, "regression_filters":()}
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
        if type(value) == dict:
            for key, val in value.items():
                if key not in self._filter_args:
                    raise KeyError(f"filter {key} is not a valid filter")
                self._filter_args[key] = val
        elif type(value) == tuple:
            self._filter_args[value[0]] = tuple(value[1])


    def construct_features(self, features: DataReader, selector: FeatureSelection, feature_range: list[int], 
                           plot: bool=True, plot_num_features: int=20, rfe_step: int=30):
        """
        Perform feature selection using a holdout strategy.

        Parameters
        ----------
        features : DataReader
            The data reader object containing the feature data.
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
        
        X_train, X_test, Y_train, Y_test = train_test_split(features.features, features.label, test_size=self.test_size, 
                                                                random_state=self.seed, stratify=features.label)
        transformed_x = features.scale(X_train)
        feature_dict = selector.generate_features(self.filter_args, transformed_x, Y_train, 
                                            feature_range, features.features, rfe_step, plot,
                                            plot_num_features)
        selector._write_dict(feature_dict)
        return feature_dict


class FeatureRegression:
    """
    A Class to perform feature selection on regression problems with a set of predefined filter methods.

    Parameters
    ----------
    seed : int
        The random seed to use for reproducibility.
    num_filters : int, optional
        The number of filter algorithms to use for feature selection. Defaults to 10.
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
    def __init__(self, seed: int, num_filters: int=4, test_size=0.2):
        self.seed = seed
        random.seed(self.seed)
        filter_names = ("SpearmanCorr", "PearsonCorr", "KendallCorr", "FechnerCorr")
        filter_names = random.sample(filter_names, num_filters)
        regression_filters = ("mutual_info", "Fscore")
        self._filter_args = {"filter_names": filter_names, "multivariate": (), "xgboost": xgb.XGBRegressor, 
                             "RFE": Ridge, "random_forest": rfr,
                            "filter_unsupervised": (), "regression_filters": regression_filters}
        self.test_size = test_size

    @property
    def filter_args(self):
        return self._filter_args
    
    @filter_args.setter
    def filter_args(self, value: tuple[str, Iterable[str]] | dict[str, Iterable[str]]):
        if isinstance(value, dict):
            for key, val in value.items():
                if key not in self._filter_args:
                    raise KeyError(f"filter {key} is not a valid filter")
                self._filter_args[key] = val
        elif isinstance(value, tuple):
            self._filter_args[value[0]] = tuple(value[1])

    def construct_features(self, features: DataReader, selector: FeatureSelection, 
                           feature_range: list[int], plot: bool=True, plot_num_features:int=20, 
                           rfe_step:int=30) -> None:
        """
        Perform feature selection using a holdout strategy.

        Parameters
        ----------
        features : DataReader
            The data reader object containing the feature data.
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

        X_train, X_test, Y_train, Y_test = train_test_split(features.features, features.label, test_size=self.test_size, 
                                                            random_state=self.seed)
        transformed_x = features.scale(X_train)
        feature_dict = selector.generate_features(self.filter_args, transformed_x, Y_train, 
                                    feature_range, features.features, rfe_step, plot, plot_num_features)
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
        num_features_min = len(features.columns) // 10
        if not num_features_max:
            num_features_max = len(features.columns) // 2 + 1
        if not step_range:
            step_range = (num_features_max - num_features_min) // 4
        feature_range = list(range(num_features_min, num_features_max, step_range))
    elif num_features_min and step_range and num_features_max:
        feature_range = list(range(num_features_min, num_features_max, step_range))
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
    features, label, variance_threshold, feature_range, num_thread, scaler, excel_file, kfold, rfe_steps, plot, \
        plot_num_features, num_filters, seed, strategy, problem = arg_parse()
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])

    num_features_min, num_features_max, step = translate_range_str_to_list(feature_range)
    

    # generate the dataset
    training_data = DataReader(label, features, variance_threshold, scaler)
    feature_range = get_range_features(training_data.features, num_features_min, num_features_max, step)
    filters = FeatureSelection(excel_file, num_thread, seed)
    # select features
    if problem == "classification":
        selection = FeatureClassification(filters.seed, num_filters, test_size)
    elif problem == "regression":
        selection = FeatureRegression(filters.seed, num_filters, test_size)

    selection.construct_features(training_data, filters, feature_range, rfe_steps, plot, plot_num_features)


   

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()