import argparse
import pandas as pd
from BioML.utilities import scale, analyse_composition, write_excel, Log
from pathlib import Path
from ITMO_FS.filters.univariate import select_k_best, UnivariateFilter
from ITMO_FS.filters.unsupervised import TraceRatioLaplacian
# Multiprocess instead of Multiprocessing solves the pickle problem in Windows (might be different in linux)
# but it has its own errors. Use multiprocessing.get_context('fork') seems to solve the problem but only available
# in Unix. Altough now it seems that with fork the programme can hang indefinitely so use spaw instead
# https://medium.com/devopss-hole/python-multiprocessing-pickle-issue-e2d35ccf96a9
import xgboost as xgb
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from ITMO_FS.filters.multivariate import STIR, TraceRatioFisher
from collections import defaultdict
import shap
import numpy as np
import matplotlib.pyplot as plt
import random
from multiprocessing import get_context  # https://pythonspeed.com/articles/python-multiprocessing/
import time
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Iterable

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


class FeatureSelection:
    """
    Class for selecting features from a dataset using different filter methods.
    
    Parameters:
    
    label: Path to label file or label column in feature dataframe. 
    
    excel_file: Path to output Excel file to save selected features.
    
    features: Path to input feature CSV file or Pandas DataFrame.
    
    variance_thres: Features with variance below this will be removed.
    
    num_thread: Number of threads for parallel execution. 
    
    scaler: Name of scaler to use before feature selection.
    
    num_split: Number of folds for KFold cross-validation.
    
    test_size: Test set size for HoldOut validation.  
    
    num_filters: Number of filters to apply for feature selection.
    
    seed: Random seed for reproducibility.
    
    Attributes:
    
    features: Input feature DataFrame.
    
    label: Target labels Series.
    
    variance_thres: Variance threshold for filtering. 
    
    num_thread: Threads for parallelism.
    
    scaler: Chosen scaler.
    
    num_splits: Number of KFold splits. 
    
    test_size: HoldOut test set size.
    
    excel_file: Output Excel file path.
    
    num_filters: Number of filters to apply.
    
    seed: Random seed.
    
    Methods:
    
    preprocess: Remove low variance features.
    
    parallel_filter: Run filters in parallel.
    
    feature_set_kfold: KFold feature selection.
    
    feature_set_holdout: HoldOut feature selection.
    
    """
    def __init__(self, label: pd.Series | str | Iterable[int|float], excel_file: str | Path,
                 features: pd.DataFrame | str | list | np.ndarray ="training_features/every_features.csv", variance_thres: int | None =0,
                 num_thread: int =10, scaler: str="robust", num_split: int=5, test_size: float=0.2, 
                 num_filters: int=10, seed: int | None=None):
        """
        Initialize a new instance of the FeatureSelection class.

        Parameters
        ----------
        label : pd.Series or str or Iterable[int or float]
            The label data for the model.
        excel_file : str or Path
            The path to the Excel file to save the selected features.
        features : pd.DataFrame or str or Iterable[list or np.ndarray]
            The input data for the model. Defaults to "training_features/every_features.csv".
        variance_thres : int, optional
            The variance threshold for feature selection. Defaults to 0.
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
        self.features, self.label = self._read_features_labels(features, label)
        analyse_composition(self.features)
        self.preprocess
        self._check_label("labels_corrected.csv") 
        self.variance_thres = variance_thres
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
    
    def _read_features_labels(self, features: str | pd.DataFrame | list | np.ndarray, 
                             labels: str | pd.Series | Iterable) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fix the feature and label data to ensure they are in the correct format.

        Parameters
        ----------
        features : str or pd.DataFrame or list or np.ndarray
            The feature data for the model.
        labels : pd.Series or str or list or set or np.ndarray
            The label data for the model.

        Raises
        ------
        TypeError
            If the features or labels are not in a valid format.

        Returns
        -------
        pd.DataFrame, pd.Series
            The fixed feature and label data.
        """
        if isinstance(features, str) and features.endswith(".csv"):
            features = pd.read_csv(f"{features}", index_col=0) # the first column shoudl contain the sample names
        elif isinstance(features, pd.DataFrame):
            features = features
        elif isinstance(features, (list, np.ndarray)):
            features = pd.DataFrame(features)
        else:
            self.log.error("features should be a csv file, an array or a pandas DataFrame")
            raise TypeError("features should be a csv file, an array or a pandas DataFrame")
        
        if isinstance(labels, pd.Series):
            label = labels
        elif isinstance(labels, (list, set, np.ndarray)):
            label = pd.Series(labels, index=features.index, name="target")

        elif isinstance(labels, str):
            if Path(labels).exists():
                label = pd.read_csv(labels, index_col=0)
            
            elif labels in features.columns:
                label = features[labels]
                features.drop(labels, axis=1, inplace=True)
        else:
            self.log.error("label should be a csv file, a pandas Series, an array or inside features")
            raise TypeError("label should be a csv file, a pandas Series, an array or inside features")

        return features, label

    def preprocess(self) -> pd.DataFrame:
        """
        Eliminate low variance features using the VarianceThreshold from sklearn
        """
        if self.variance_thres is not None:
            variance = VarianceThreshold(self.variance_thres)
            fit = variance.fit_transform(self.features)
            self.features = pd.DataFrame(fit, index=self.features.index, 
                                         columns=variance.get_feature_names_out())

        return self.features

    @staticmethod
    def univariate(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                   num_features: int, feature_names: Iterable[str], filter_name: str) -> pd.Series:
        """
        Perform univariate feature selection.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            The training feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        num_features : int
            The number of features or columns to select.
        feature_names : Iterable[str]
            The names of the features or columns.
        filter_name : str
            The name of the statistical filter to use.

        Returns
        -------
        pd.Series
            A series containing the feature scores, sorted in descending order.
        """
        ufilter = UnivariateFilter(filter_name, select_k_best(num_features))
        ufilter.fit(X_train, Y_train)
        scores = {x: v for x, v in zip(feature_names, ufilter.feature_scores_)}
        # sorting the features
        if filter_name != "LaplacianScore":
            if filter_name in ["SpearmanCorr", "PearsonCorr", "KendallCorr", "FechnerCorr"]:
                scores = pd.Series(dict(sorted(scores.items(), key=lambda items: abs(items[1]), reverse=True)))
            else:
                scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
        else:
            scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1])))
        return scores

    @staticmethod
    def multivariate(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                     num_features: int, feature_names: Iterable[str], filter_name: str) -> pd.Series:
        """
        Perform multivariate feature selection.

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
        filter_name : str
            The name of the statistical filter to use.

        Returns
        -------
        pd.Series
            A series containing the feature scores, sorted in descending order.
        """
        if filter_name == "STIR":
            ufilter = STIR(num_features, k=5).fit(X_train, Y_train)
            scores = {x: v for x, v in zip(feature_names, ufilter.feature_scores_)}
        else:
            ufilter = TraceRatioFisher(num_features).fit(X_train, Y_train)
            scores = {x: v for x, v in zip(feature_names, ufilter.score_)}
        scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
        return scores

    @staticmethod
    def unsupervised(X_train: pd.DataFrame | np.ndarray, num_features: int, 
                     feature_names: Iterable[str], filter_name: str) -> pd.Series:
        """
        Perform unsupervised feature selection using a statistical filter.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            The training feature data.
        num_features : int
            The number of features to select.
        feature_names : list or np.ndarray
            The names of the features.
        filter_name : str
            The name of the statistical filter to use.

        Returns
        -------
        pd.Series
            A series containing the feature scores, sorted in descending order.
        """
        if "Trace" in filter_name:
            ufilter = TraceRatioLaplacian(num_features).fit(X_train)
            scores = {x: v for x, v in zip(feature_names, ufilter.score_)}

        scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))

        return scores

    def _get_num_feature_range(self, num_features_min: int | None=None, 
                               num_features_max: int | None=None, step_range: int | None=None) -> list:
        """
        Get a range of numbers for the number of features to select.

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
            num_feature_range = list(range(num_features_min, num_features_max, step_range))
        elif num_features_min and step_range and num_features_max:
            num_feature_range = list(range(num_features_min, num_features_max, step_range))
        else:
            num_feature_range = [num_features_min]
    
        return num_feature_range
    
    def random_forest(self, X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray,
                      feature_names: Iterable[str], problem: str="classification") -> pd.Series:
        """
        Perform feature selection using a random forest model.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            The training feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        feature_names : list or np.ndarray
            The names of the features.
        problem : str, optional
            The type of problem to solve. Defaults to "classification".

        Returns
        -------
        pd.Series
            A series containing the feature importances, sorted in descending order.
        """
        if problem == "classification":
            forest_model = rfc(class_weight="balanced_subsample", random_state=self.seed, max_features=0.7, max_samples=0.8,
                         min_samples_split=6, n_estimators=200, n_jobs=self.num_thread,
                         min_impurity_decrease=0.1)
        else:
            forest_model = rfr(random_state=self.seed, max_features=0.7, max_samples=0.8,
                         min_samples_split=6, n_estimators=200, n_jobs=self.num_thread, 
                         min_impurity_decrease=0.1)
        forest_model.fit(X_train, Y_train)
        gini_importance = pd.Series(forest_model.feature_importances_, index=feature_names)
        gini_importance.sort_values(ascending=False, inplace=True)

        return gini_importance

    def xgbtree(self, X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                feature_names: Iterable[str], split_ind: int, plot: bool=True, plot_num_features: int=20,
                problem: str="classification") -> pd.Series:
        """
        Perform feature selection using a xgboost model.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            The training feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        feature_names : list or np.ndarray
            The names of the features.
        split_ind : int
            The index of the current split.
        plot : bool, optional
            Whether to plot the feature importances. Defaults to True.
        plot_num_features : int, optional
            The number of features to include in the plot. Defaults to 20.
        problem : str, optional
            The type of problem to solve. Defaults to "classification".

        Returns
        -------
        pd.Series
            A series containing the feature importances, sorted in descending order.
        """
        if problem == "classification":
            XGBOOST = xgb.XGBClassifier(learning_rate=0.01, n_estimators=200, max_depth=4, gamma=0,
                                    subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                    nthread=self.num_thread, seed=self.seed)
        else:
            XGBOOST = xgb.XGBRegressor(learning_rate=0.01, n_estimators=200, max_depth=4, gamma=0,
                                    subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
                                    nthread=self.num_thread, seed=self.seed)
        # Train the model
        XGBOOST.fit(X_train, Y_train)
        important_features = XGBOOST.get_booster().get_score(importance_type='gain')
        important_features = pd.Series(important_features)
        important_features.sort_values(ascending=False, inplace=True)
        xgboost_explainer = shap.TreeExplainer(XGBOOST, X_train, feature_names=feature_names)
        shap_values = xgboost_explainer.shap_values(X_train, Y_train)
        shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), feature_names).sort_values(ascending=False)
        shap_importance = shap_importance.loc[lambda x: x > 0]

        if plot:
            shap_dir = (self.excel_file.parent / "shap_features")
            shap_dir.mkdir(parents=True, exist_ok=True)
            shap_importance.to_csv(shap_dir / f"shap_importance_kfold{split_ind}.csv")
            shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type='bar', show=False,
                              max_display=plot_num_features)
            plt.savefig(shap_dir / f"shap_kfold{split_ind}_top_{plot_num_features}_features.png", dpi=800)
            shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False,
                              max_display=plot_num_features)
            plt.savefig(shap_dir / f"feature_influence_on_model_prediction_kfold{split_ind}.png", dpi=800)
        return shap_importance

    def rfe_linear(self, X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, num_features: int, 
                   feature_names: Iterable[str], step: int=30, problem: str="classification") -> list[str]:
        """
        Perform feature selection using recursive feature elimination with a linear model.

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
        step : int, optional
            The number of features to remove at each iteration. Defaults to 30.
        problem : str, optional
            The type of problem to solve. Defaults to "classification".

        Returns
        -------
        list
            A list of the selected feature names.
        """
        if problem == "classification":
            linear_model = RidgeClassifier(random_state=self.seed, alpha=4)  
        else:
            linear_model = Ridge(random_state=self.seed, alpha=4)
        rfe = RFE(estimator=linear_model, n_features_to_select=num_features, step=step)
        rfe.fit(X_train, Y_train)
        features = rfe.get_feature_names_out(feature_names)
        return features
    
    @staticmethod
    def regression_filters(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, feature_nums: int, 
                       feature_names: Iterable[str], reg_func: str) -> pd.Series:
        """
        Perform feature selection using regression filters.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            The training feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        feature_nums : int
            The number of features to select.
        feature_names : list or np.ndarray
            The names of the features.
        reg_func : str
            The name of the regression filter to use. Available options are "mutual_info" and "Fscore".

        Returns
        -------
        pd.Series
            A series containing the feature scores, sorted in descending order.
        """
        reg_filters = {"mutual_info": mutual_info_regression, "Fscore":f_regression}
        sel = SelectKBest(reg_filters[reg_func], k=feature_nums)
        sel.fit(X_train, Y_train)
        scores = sel.scores_
        scores = {x: v for x, v in zip(feature_names, scores)}
        scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
        return scores
    
    def parallel_filter(self, X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, num_features: int, 
                        feature_names: Iterable[str], split_ind: int, plot: bool=True, plot_num_features: int=20, 
                        problem: str="classification", filter_args: dict={"filter_names":(), "multivariate":(), 
                        "filter_unsupervised":(), "regression_filters":()}) -> pd.Series:
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
        split_ind : int
            The index of the current split.
        plot : bool, optional
            Whether to plot the feature importances. Defaults to True.
        plot_num_features : int, optional
            The number of features to include in the plot. Defaults to 20.
        problem : str, optional
            The type of problem to solve. Defaults to "classification".
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
            for num, res in enumerate(pool.starmap(self.univariate, arg_univariate)):
                print(f"univariate filter: {filter_names[num]}")
                results[filter_names[num]] = res

            for num, res in enumerate(pool.starmap(self.multivariate, arg_multivariate)):
                print(f"multivariate filter: {multivariate[num]}")
                results[multivariate[num]] = res

            for num, res in enumerate(pool.starmap(self.unsupervised, arg_unsupervised)):
                print(f"unsupervised filter: {filter_unsupervised[num]}")
                results[filter_unsupervised[num]] = res
            
            for num, res in enumerate(pool.starmap(self.regression_filters, arg_regression)):
                print(f"regression filter: {regression_filters[num]}")
                results[regression_filters[num]] = res
    
        results["xgbtree"] = self.xgbtree(X_train, Y_train, feature_names, split_ind, plot, 
                                          plot_num_features, problem)
        results["random"] = self.random_forest(X_train, Y_train, feature_names, problem)

        return pd.concat(results)
    
    def _construct_features(self, univariate_features: pd.DataFrame, features: pd.DataFrame, feature_dict: dict, 
                        num_feature_range: Iterable[int], transformed: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                        split_ind: int, rfe_step: int=30, problem: str="classification") -> None:
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
        num_feature_range : list or np.ndarray
            A range of the number of features to include in each feature set.
        transformed : pd.DataFrame
            The transformed feature data.
        Y_train : pd.Series or np.ndarray
            The training label data.
        split_ind : int
            The index of the current split.
        rfe_step : int, optional
            The number of features to remove at each iteration in recursive feature elimination. Defaults to 30.
        problem : str, optional
            The type of problem to solve. Defaults to "classification".

        Returns
        -------
        None
        """

        for num_features in num_feature_range:
            print(f"generating a feature set of {num_features} dimensions")
            for filters in univariate_features.index.unique(0):
                feat = univariate_features.loc[filters]
                feature_dict[f"{filters}_{num_features}"][f"split_{split_ind}"] = features[feat.index[:num_features]]
            rfe_results = self.rfe_linear(transformed, Y_train, num_features, features.columns, rfe_step, problem)
            feature_dict[f"rfe_{num_features}"][f"split_{split_ind}"] = features[rfe_results]

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

    def feature_set_kfold(self, filter_args: dict, num_features_min: int=None, num_features_max: int=None, 
                      step_range: Iterable[int]=None, rfe_step: int=30, plot: bool=True, 
                      plot_num_features: int=20, problem: str="classification") -> None:
        """
        Perform feature selection using k-fold cross-validation.

        Parameters
        ----------
        filter_args : dict
            A dictionary containing the arguments for each filter algorithm.
        num_features_min : int, optional
            The minimum number of features to include in each feature set. Defaults to None.
        num_features_max : int, optional
            The maximum number of features to include in each feature set. Defaults to None.
        step_range : list or np.ndarray, optional
            A range of the number of features to include in each feature set. Defaults to None.
        rfe_step : int, optional
            The number of features to remove at each iteration in recursive feature elimination. Defaults to 30.
        plot : bool, optional
            Whether to plot the feature importances. Defaults to True.
        plot_num_features : int, optional
            The number of features to include in the plot. Defaults to 20.
        problem : str, optional
            The type of problem to solve. Defaults to "classification".

        Returns
        -------
        None
        """
        feature_dict = defaultdict(dict)
        if problem == "classification":
            skf = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.seed)
        else:
            skf = ShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.seed)

        num_feature_range = self._get_num_feature_range(num_features_min, num_features_max, step_range)

        for i, (train_index, test_index) in enumerate(skf.split(self.features, self.label)):
            self.log.info(f"kfold {i}")
            self.log.info("------------------------------------")
            X_train = self.features.iloc[train_index]
            Y_train = self.label.iloc[train_index].values.ravel()
            transformed, scaler_dict = scale(self.scaler, X_train)
            # for each split I do again feature selection and save all the features selected from different splits
            # but with the same selector in the same dictionary
            self.log.info("filtering the features")
            ordered_features = self.parallel_filter(transformed, Y_train, num_feature_range[-1], self.features.columns, i,
                                                    plot, plot_num_features, problem, filter_args)
            self._construct_features(ordered_features, self.features, feature_dict, num_feature_range, transformed, Y_train, 
                                     i, rfe_step, problem)

        self._write_dict(feature_dict)

    def feature_set_holdout(self, filter_args: dict, num_features_min: int=None, num_features_max: int=None, 
                            step_range: Iterable[int]=None, plot: bool=True, plot_num_features: int=20, rfe_step: int=30, 
                            problem: str="classification") -> None:
        """
        Perform feature selection using a holdout set.

        Parameters
        ----------
        filter_args : dict
            A dictionary containing the arguments for each filter.
        num_features_min : int, optional
            The minimum number of features to include in each feature set. Defaults to None.
        num_features_max : int, optional
            The maximum number of features to include in each feature set. Defaults to None.
        step_range : list or np.ndarray, optional
            A range of the number of features to include in each feature set. Defaults to None.
        plot : bool, optional
            Whether to plot the feature importances. Defaults to True.
        plot_num_features : int, optional
            The number of features to include in the plot. Defaults to 20.
        rfe_step : int, optional
            The number of features to remove at each iteration in recursive feature elimination. Defaults to 30.
        problem : str, optional
            The type of problem to solve. Defaults to "classification".

        Returns
        -------
        None
        """
        feature_dict = defaultdict(dict)
        num_feature_range = self._get_num_feature_range(num_features_min, num_features_max, step_range)
        if problem == "classification":
            X_train, X_test, Y_train, Y_test = train_test_split(self.features, self.label, test_size=self.test_size, 
                                                                random_state=self.seed, stratify=self.label)
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(self.features, self.label, test_size=self.test_size, random_state=self.seed)
        
        transformed, scaler_dict = scale(self.scaler, X_train)
        print("filtering the features")
        ordered_features = self.parallel_filter(transformed, Y_train, num_feature_range[-1], self.features.columns, 0,
                                                plot, plot_num_features, problem, filter_args)
        self._construct_features(ordered_features, self.features, feature_dict, num_feature_range, transformed, Y_train, 
                                 0, rfe_step, problem)
        
        self._write_dict(feature_dict)


class FeatureClassification(FeatureSelection):
    def __init__(self, label, excel_file, features="training_features/every_features.csv", variance_thres=0, num_thread=10, 
                 scaler="robust", num_split=5, test_size=0.2, num_filters=10, seed=None):
        super().__init__(label, excel_file, features, variance_thres, num_thread, scaler, num_split, test_size, num_filters, seed)
        """Subclass to perform feature selection on classification problems with a set of predefined filter methods"""
        random.seed(self.seed)
        filter_names = ("FRatio", "SymmetricUncertainty", "SpearmanCorr", "PearsonCorr", "Chi2", "Anova",
                        "LaplacianScore", "InformationGain", "KendallCorr", "FechnerCorr")
        filter_names = random.sample(filter_names, self.num_filters)
        multivariate = ("STIR", "TraceRatioFisher")
        filter_unsupervised = ("TraceRatioLaplacian",)
        self._filter_args = {"filter_names": filter_names, "multivariate": multivariate, 
                            "filter_unsupervised": filter_unsupervised, "regression_filters":()}

        self.log.info("Classification Problem")
        self.log.info(f"Using {len(filter_names)+len(multivariate)+len(filter_unsupervised)} filters: {filter_names}, {filter_unsupervised} and {multivariate}")

    @property
    def filter_args(self):
        return self._filter_args
    
    @filter_args.setter
    def filter_args(self, value: tuple[str, Iterable[str]]):
        self._filter_args[value[0]] = tuple(value[1])

    def construct_kfold_classification(self, num_features_min=None, num_features_max=None, step_range=None, rfe_step=30,
                                        plot=True, plot_num_features=20):
        
        self.feature_set_kfold(self.filter_args, num_features_min, num_features_max, 
                               step_range, rfe_step, plot, plot_num_features)
        

    def construct_holdout_classification(self, num_features_min=None, num_features_max=None, step_range=None,
                                      plot=True, plot_num_features=20, rfe_step=30):
        
        self.feature_set_holdout(self.filter_args, num_features_min, num_features_max, step_range,
                                 plot, plot_num_features, rfe_step)


class FeatureRegression(FeatureSelection):
    def __init__(self, label, excel_file, features="training_features/every_features.csv", variance_thres=0, num_thread=10, 
                 scaler="robust", num_split=5, test_size=0.2, num_filters=4, seed=None):
        super().__init__(label, excel_file, features, variance_thres, 
                         num_thread, scaler, num_split, test_size, 
                         num_filters, seed)
        
        """Subclass to perform feature selection on regression problems with a set of predefined filter methods"""
        random.seed(self.seed)
        filter_names = ("SpearmanCorr", "PearsonCorr", "KendallCorr", "FechnerCorr")
        filter_names = random.sample(filter_names, self.num_filters)
        regression_filters = ("mutual_info", "Fscore")
        self._filter_args = {"filter_names": filter_names, "multivariate": (), 
                            "filter_unsupervised": (), "regression_filters": regression_filters}
        self.log.info("Regression Problem")
        self.log.info(f"Using {len(filter_names) + len(regression_filters)} filters: {filter_names} and {regression_filters}")

    @property
    def filter_args(self):
        return self._filter_args
    
    @filter_args.setter
    def filter_args(self, value: tuple[str, Iterable[str]]):
        self._filter_args[value[0]] = tuple(value[1])

    def construct_kfold_regression(self, num_features_min=None, num_features_max=None, step_range=None, rfe_step=30,
                                  plot=True, plot_num_features=20):
         
         self.feature_set_kfold(self.filter_args, num_features_min, num_features_max, 
                               step_range, rfe_step, plot, plot_num_features, problem="regression")

    def construct_holdout_regression(self, num_features_min=None, num_features_max=None, step_range=None,
                                      plot=True, plot_num_features=20, rfe_step=30):
        
        self.feature_set_holdout(self.filter_args, num_features_min, num_features_max, step_range,
                                 plot, plot_num_features, rfe_step, problem="regression")


def main():
    features, label, variance_threshold, feature_range, num_thread, scaler, excel_file, kfold, rfe_steps, plot, \
        plot_num_features, num_filters, seed, strategy, problem = arg_parse()
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])
    feature_range = feature_range.split(":")
    num_features_min, num_features_max, step = feature_range
    if num_features_min == "none":
        num_features_min = None
        step = None
        num_features_max = None
    else:
        num_features_min = int(num_features_min)
        if step.isdigt():
            step = int(step)
        else:
            step = None
        if num_features_max.isdigit():
            num_features_min = int(num_features_max)
        else:
            num_features_max = None
    if problem == "classification":
        selection = FeatureClassification(label, excel_file, features, variance_threshold, num_thread, scaler,
                                    num_split, test_size, num_filters, seed)
        if strategy == "holdout":
            selection.construct_holdout_classification(num_features_min, num_features_max, step, rfe_steps, plot, plot_num_features)
        elif strategy == "kfold":
            selection.construct_kfold_classification(num_features_min, num_features_max, step, rfe_steps, plot, plot_num_features)
    else:
        selection = FeatureRegression(label, excel_file, features, variance_threshold, num_thread, scaler,
                                    num_split, test_size, num_filters, seed)
        if strategy == "holdout":
            selection.construct_holdout_regression(num_features_min, num_features_max, step, rfe_steps, plot, plot_num_features)
        elif strategy == "kfold":
            selection.construct_kfold_regression(num_features_min, num_features_max, step, rfe_steps, plot, plot_num_features)

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()