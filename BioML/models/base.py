from dataclasses import dataclass, field
import pandas as pd
from typing import Callable, Any, Iterator
import numpy as np
from pathlib import Path
import time
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from sklearn.metrics import average_precision_score
from typing import Iterable, Sequence
import warnings
from collections import defaultdict
from typing import Protocol
from ..utilities.utils import Log, write_results
from ..utilities.custom_errors import DifferentLabelFeatureIndexError
from .metrics import ndcg_at_k, pear_r


class ModelArguments(Protocol):
    """
    It defines the methods and the attributes of a class.

    Parameters
    ----------
    Protocol : [type]
        _description_
    """
    drop: Iterable[str]
    selected: Iterable[str]
    optimize: str
    test_size: float
    plot: tuple[str, ...]
    ranking_params: dict[str, float]
    add: Iterable[Any]

    def _calculate_score_dataframe(self, dataframe: pd.DataFrame) -> int | float:
        ...

    def sort_holdout_prediction(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass(slots=True)
class DataParser:
    """
    A class for parsing feature and label data.

    Parameters
    ----------
    features : pd.DataFrame or str or list or np.ndarray
        The feature data.
    label : pd.Series or str or Iterable[int|float] or None, optional
        The label data. Defaults to None.
    outliers : Iterable[str], optional
        An iterable containing the indices of the outliers in the feature and label data. Defaults to an empty ().
    sheets : str or int, optional
        The sheet name or index to read from an Excel file. Defaults to 0.
    """
    features: pd.DataFrame | str | list | np.ndarray
    label: pd.Series | pd.DataFrame | str | Iterable[int|float|str] | None = None
    outliers: Iterable[str] = ()
    sheets: str | int | None = None

    def __post_init__(self):
        self.features = self.read_features(self.features, sheets=self.sheets)
        if self.label is not None:
            self.label = self.read_labels(self.label) # type: ignore
            if not isinstance(self.label, str):
                if len(self.features.index.difference(self.label.index)):
                    raise DifferentLabelFeatureIndexError("The label and feature indices are different")
                self.features = pd.concat([self.features, self.label.rename("target")], axis=1)
                self.label = "target"
        
        if self.outliers:
            self.features = self.remove_outliers(self.features, self.outliers)

    @classmethod
    def read_features(cls, features: str | pd.DataFrame | list | np.ndarray, sheets: str | int | None=None) -> pd.DataFrame:
        """
        Reads the feature data from a file or returns the input data.

        Parameters
        ----------
        features : str or pd.DataFrame or list or np.ndarray
            The feature data.
        sheets : str or int, optional
            The sheet name or index to read from an Excel file. Defaults to None.

        Returns
        -------
        pd.DataFrame
            The feature data as a pandas DataFrame.
        """
        # concatenate features and labels
        match features:
            case str() | Path() as feature if str(feature).endswith(".csv"):
                return pd.read_csv(f"{features}", index_col=0) # the first column should contain the sample names
            
            case  str() | Path() as feature if str(feature).endswith(".xlsx"):
                sheets = sheets if sheets else 0
                with pd.ExcelFile(features) as file:
                    if len(file.sheet_names) > 1:
                        warnings.warn(f"The excel file contains more than one sheet, only the sheet {sheets} will be used")
                return pd.read_excel(features, index_col=0, engine='openpyxl', sheet_name=sheets)
            
            case pd.DataFrame() as feature:
                return feature
            
            case list() | np.ndarray() as feature:
                return pd.DataFrame(feature)

            case _:
                raise ValueError(f"features should be a csv or excel file, an array or a pandas DataFrame, you provided {features}")
        
    def read_labels(self, label: str | pd.Series) -> str | pd.Series:
        """
        Reads the label data from a file or returns the input data.

        Parameters
        ----------
        label : str or pd.Series
            The label data.

        Returns
        -------
        pd.Series
            The label data as a pandas Series.
        """
        match label:
            case pd.Series() as labels:
                return labels
            
            case pd.DataFrame() as labels:
                return labels.squeeze()
            
            case str() | Path() as labels if Path(labels).exists() and Path(labels).suffix == ".csv":
                labels = pd.read_csv(labels, index_col=0)
                return labels.squeeze()
            
            case str(labels) if labels in self.features.columns:
                return labels
            
            case list() | np.ndarray() as labels:
                return pd.Series(labels, index=self.features.index, name="target")
            
            case _:
                raise ValueError(f"label should be a csv file, an array, a pandas Series, DataFrame or inside features: you provided {label}")
    
    @classmethod
    def remove_outliers(cls, training_features: pd.DataFrame, outliers: Iterable[str]):
        """
        Remove outliers from the train data

        Parameters
        ----------
        training_features : pd.DataFrame
            The training data.

        outliers: Iterable[str]
            An iterable containing the indices to remove from the training set
            
        Returns
        -------
        pd.DataFrame
            The data with outliers removed.
        """
        #remove outliers
        training_features = training_features.loc[[x for x in training_features.index if x not in outliers]]

        return training_features
    
    def drop(self)-> pd.DataFrame:
        """
        Return the feature without the labels in it

        Returns
        -------
        pd.DataFrame
            The training feature without the label data, 
        """
        return self.features.drop(columns=self.label)
    
    def __repr__(self):
        string = f"""Data with:\n    num. samples: {len(self.features)}\n    num. columns: {len(self.features.columns)}\n    name label column: {self.label}\n    num. outliers: {len(self.outliers)}"""
        return string


@dataclass
class PycaretInterface:
    """
    A class for interfacing with the PyCaret library for machine learning. 
    Learn more here:
    https://pycaret.gitbook.io/docs/
    https://medium.com/analytics-vidhya/pycaret-101-for-beginners-27d9aefd34c5
    https://towardsdatascience.com/5-things-you-are-doing-wrong-in-pycaret-e01981575d2a
    """
    objective: str
    seed: None | int = None
    optimize: str = "MCC"
    #verbose: bool = False
    scaler: str = "zscore"
    budget_time: None | int = None
    log = Log("model_training")
    best_model: int = 3
    output_path: Path | str | None= None
    experiment_name: str | None = None
    log_experiment: bool = True
    # No need to provide values
    _plots: Sequence[str] = field(init=False)
    _final_models: Sequence[str] = field(init=False)
    original_plots: list[str] = field(init=False)
    original_models: list[str] = field(init=False)
    model: ClassificationExperiment | RegressionExperiment = field(init=False, repr=False, compare=False)
    
    def __post_init__(self):
        configuration = {"classification": 
                            {"plots": ["confusion_matrix", "learning", "class_report", "auc", "pr"],
                             "models": ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 
                                                    'mlp', 'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 
                                                     'lightgbm', 'catboost', 'dummy'],
                            "pycaret": ClassificationExperiment},
                        "regression": 
                            {"plots": ["residuals", "error", "learning"],
                            "models": ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 
                                   'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 
                                   'lightgbm', 'catboost', 'dummy'],
                            "pycaret": RegressionExperiment}}

        self.pycaret = configuration[self.objective]["pycaret"]()
        self._plots = configuration[self.objective]["plots"]
        self._final_models = configuration[self.objective]["models"]

        self.experiment_name = self.objective.capitalize() if self.experiment_name is None else self.experiment_name
        self.original_plots = self._plots.copy()
        self.original_models = self._final_models.copy()
        if not self.seed: 
            self.seed = int(time.time())
        if isinstance(self.budget_time, (int, float)):
            if self.budget_time < 0:
                raise ValueError("The budget time should be greater than 0")
        elif self.budget_time is not None:
            raise ValueError("The budget time should be greater than 0 or None")
        
        self.output_path: Path = Path.cwd() if self.output_path is None else Path(self.output_path)
        
        self.log.info("------------------------------------------------------------------------------")
        self.log.info("PycaretInterface parameters")
        self.log.info(f"Seed: {self.seed}")
        self.log.info(f"Budget time: {self.budget_time}")
        self.log.info(f"The number of models to select: {self.best_model}")
        self.log.info(f"Output path: {self.output_path}")
    
    @staticmethod
    def _check_value(value: str | Sequence[str] | Any, element_list: list[str], element_name: str) -> Sequence[str]:
        """
        To set the value of the plots and models

        Parameters
        ----------
        value : str | Iterable[str]
            The model or plot names to check
        element_list : list[str]
            The list to check against
        element_name : str
            If it is plot or model

        Returns
        -------
        list[str]
            The list of the models or plots

        Raises
        ------
        ValueError
            If the value is not in the element_list
        ValueError
            If the value is not a string or an array of strings
        """
        if not len(value):
            return []
        if isinstance(value, (list, np.ndarray, tuple, set)):
            if isinstance(value[0], str):
                test = list(set(value).difference(element_list))
                for x in test:
                    if isinstance(x, str):
                        raise ValueError(f"the {element_name} should be one of the following: {element_list} and not {test}")
        elif isinstance(value, str):
            if value not in element_list:
                raise ValueError(f"the {element_name} should be one of the following: {element_list} and not {value}")
            value = [value]
        
        else:
            value = [value]
        return value

    @property
    def plots(self) -> Sequence[str]:
        """
        The plots that should be saved
        """
        return self._plots
    
    @plots.setter
    def plots(self, value: str | Sequence[str]):
        
        self._plots = self._check_value(value, self.original_plots, "plots")

    @property
    def final_models(self) -> Sequence[str]:
        """
        The models to be used for classification or regression use one of the keys, 
        you can include custom models
        just pass an untrained model object consistent with scikit-learn API (fit, predict, predict_proba, etc.)

        ---------------Classification models---------------
        {'lr': 'Logistic Regression',
        'knn': 'K Neighbors Classifier',
        'nb': 'Naive Bayes',
        'dt': 'Decision Tree Classifier',
        'svm': 'SVM - Linear Kernel',
        'rbfsvm': 'SVM - Radial Kernel',
        'gpc': 'Gaussian Process Classifier',
        'mlp': 'MLP Classifier',
        'ridge': 'Ridge Classifier',
        'rf': 'Random Forest Classifier',
        'qda': 'Quadratic Discriminant Analysis',
        'ada': 'Ada Boost Classifier',
        'gbc': 'Gradient Boosting Classifier',
        'lda': 'Linear Discriminant Analysis',
        'et': 'Extra Trees Classifier',
        'xgboost': 'Extreme Gradient Boosting',
        'lightgbm': 'Light Gradient Boosting Machine',
        'catboost': 'CatBoost Classifier',
        'dummy': 'Dummy Classifier'}
    
        -----------Regression models-----------------------
        {'lr': 'Linear Regression',
        'lasso': 'Lasso Regression',
        'ridge': 'Ridge Regression',
        'en': 'Elastic Net',
        'lar': 'Least Angle Regression',
        'llar': 'Lasso Least Angle Regression',
        'omp': 'Orthogonal Matching Pursuit',
        'br': 'Bayesian Ridge',
        'ard': 'Automatic Relevance Determination',
        'par': 'Passive Aggressive Regressor',
        'ransac': 'Random Sample Consensus',
        'tr': 'TheilSen Regressor',
        'huber': 'Huber Regressor',
        'kr': 'Kernel Ridge',
        'svm': 'Support Vector Regression',
        'knn': 'K Neighbors Regressor',
        'dt': 'Decision Tree Regressor',
        'rf': 'Random Forest Regressor',
        'et': 'Extra Trees Regressor',
        'ada': 'AdaBoost Regressor',
        'gbr': 'Gradient Boosting Regressor',
        'mlp': 'MLP Regressor',
        'xgboost': 'Extreme Gradient Boosting',
        'lightgbm': 'Light Gradient Boosting Machine',
        'catboost': 'CatBoost Regressor',
        'dummy': 'Dummy Regressor'}
        """
        return self._final_models
     
    @final_models.setter
    def final_models(self, value: str |Iterable[str]) -> None:
        self._final_models = self._check_value(value, self.original_models, "models")

    def setup_training(self, feature: pd.DataFrame, label_name: str, fold: int=5, 
                       test_size:float=0.2, k: int=8, penalty: int=2, 
                       **kwargs: Any):
        """
        Call pycaret set_up for the training.

        Parameters
        ----------
        feature : pd.DataFrame
            The training data.
        label_name: str
            The name of the target column.
        fold : int
            The number of cross-validation folds to use.
        test_size : float
            The proportion of the dataset to include in the test split.
        kwargs : dict
            Other parameters to pass to pycaret.setup.
        Returns
        -------
        self
            PycaretInterface object.
        """
        self.pycaret.setup(data=feature, target=label_name, normalize=True, preprocess=True, 
                           log_experiment=self.log_experiment, experiment_name=self.experiment_name, normalize_method=self.scaler,
                           session_id = self.seed, fold_shuffle=True, fold=fold, verbose=False, train_size=1-test_size, 
                           **kwargs)

        if self.objective == "classification":
            self.log.info("Added metrics for classification: Average precision")
            self.pycaret.add_metric("averagePre", "Average Precision Score", average_precision_score,  # type: ignore
                                   average="weighted", target="pred_proba", multiclass=False)
        elif self.objective == "regression":
            self.log.info(f"Added metrics for regression: NDCG k={k}, penalty={penalty} and Pearson")
            self.pycaret.add_metric("ndcg", "NDCG", ndcg_at_k, k=k, penalty=penalty) # type: ignore
            self.pycaret.add_metric("pearson", "Pearson", pear_r, **kwargs) # type: ignore
            
        config: pd.DataFrame = self.pycaret.pull(pop=True)
        if not (self.output_path / "config_setup_pycaret.csv").exists():
            self.output_path.mkdir(parents=True, exist_ok=True)
            config.to_csv(self.output_path / "config_setup_pycaret.csv") 
        return self

    def train(self, cross_validation: bool=True):
        """
        Train the machine learning models using all default hyperparameters so just one set of parameters
        
        Parameters
        ----------
        cross_validation : bool, optional
            If True, use cross-validation, by default True

        Returns
        -------
        Tuple[dict, dict]
            A tuple containing the results of the training and the trained models.
        """
        self.log.info("--------------------------------------------------------")
        self.log.info(f"Training {self.objective} models")
        self.log.info(f"The models used {self.final_models}")
        self.log.info(f"The number of models used {len(self.final_models)}")
        results = {}
        returned_models = {}
        if self.budget_time:
            self.log.info(f"Time budget is {self.budget_time} minutes")
        runtime_start = time.time()
        count=0
        for m in self.final_models:
            model_time_start = time.time()
            model = self.pycaret.create_model(m, return_train_score=True, verbose=False, cross_validation=cross_validation)
            model_results = self.pycaret.pull(pop=True)
            model_results = model_results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
            if not isinstance(m, str):
                try:
                    m = m.__class__.__name__
                except AttributeError:
                    m = f"custom_model_{count}"
                    count += 1
            returned_models[m] = model
            results[m] = model_results
            runtime_train = time.time()
            total_runtime = round((runtime_train - runtime_start) / 60, 3)
            model_time = round((runtime_train - model_time_start) / 60, 3)
            self.log.info(f"Model {m} trained in {model_time} minutes")
            if self.budget_time and total_runtime > self.budget_time:
                self.log.info(
                    f"Total runtime {total_runtime} is over time budget by {total_runtime - self.budget_time} minutes, breaking loop"
                )
                break

        self.log.info(f"Training over: Total runtime {total_runtime} minutes") # type: ignore

        return results, returned_models
    
    def plot_best_models(self, sorted_models: dict[str, Any], key: str | None=None):
        """
        Analyze the best models and plot them.

        Parameters
        ----------
        sorted_models : dict
            A dictionary containing the trained models sorted by performance.
        key : str, optional
            Something to separate even more. Defaults to None.
        """
        if not self.plots:
            return
        self.log.info("Analyse the best models and plotting them")
        for ind, (name, model) in enumerate(sorted_models.items(), 1):
            if ind <= self.best_model:
                self.log.info(f"Analyse the top {ind} model: {name}")
                if not key:
                    plot_path = self.output_path / "model_plots" / name # type: ignore
                else:
                    plot_path = self.output_path / "model_plots" / name / key # type: ignore
                plot_path.mkdir(parents=True, exist_ok=True)
                for pl in self.plots:
                    self.pycaret.plot_model(model, pl, save=plot_path, verbose=False)

    def get_params(self, name: str, model: Any)-> pd.Series:
        """
        Get the parameters of a trained model.

        Parameters
        ----------
        name : str
            The name of the trained model.
        model : Any
            The trained model.

        Returns
        -------
        pd.Series
            A pandas series containing the parameters of the trained model.
        """
        if name != "catboost":
            params = model.get_params()
        else:
            params = model.get_all_params()
        params = {key: value for key, value in params.items() if key not in 
                    ["warm_start", "verbose", "oob_score"]}
        params = pd.Series(params)
        return params
    
    def get_params_stacked(self, stacked_models: Any) -> pd.Series:
        """
        Get the parameters of the final estimator in a stacked ensemble model.

        Parameters
        ----------
        stacked_models : Any
            The trained stacked ensemble model.

        Returns
        -------
        pd.Series
            A pandas series containing the parameters of the final estimator.
        """
        params_dict = stacked_models.get_params()
        return pd.Series(params_dict["final_estimator"].get_params())

    def get_best_params_multiple(self, sorted_models: dict[str, Any]) -> pd.Series:
        """
        Get the best parameters for multiple models.

        Parameters
        ----------
        sorted_models : dict[str, Any]
            A dictionary containing the trained models sorted by performance.

        Returns
        -------
        pd.Series
            A pandas Series containing the best parameters for each model.
        """
        model_params = {}
        for ind, (name, model) in enumerate(sorted_models.items(), 1):
            if ind > self.best_model:
                break
            try:
                params = self.get_params(name, model)
                model_params[name] = params
            except AttributeError: # in case the model does not have get_params method
                pass
        return pd.concat(model_params) # type: ignore
    
    def retune_model(self, name: str, model: Any, num_iter: int=50, 
                     fold: int=5, custom_grid=None) -> tuple[Any, pd.DataFrame, pd.Series]:
        """
        Retune the specified model using Optuna.

        Parameters
        ----------
        name : str
            The name of the model to retune.
        model : Any
            The trained model to retune.
        num_iter : int, optional
            The number of iterations to use for tuning. Defaults to 30.
        fold : int, optional
            The number of cross-validation folds to use. Defaults to 5.

        Returns
        -------
        Tuple[Any, pd.DataFrame, pd.Series]
            A tuple containing the retuned model, the results of the tuning, and the parameters used for tuning.
        """
        self.log.info("---------Retuning the best models--------------")
        self.log.info(f"num_iter: {num_iter}")
        self.log.info(f"fold: {fold}")
        tuned_model = self.pycaret.tune_model(model, optimize=self.optimize, search_library="optuna", search_algorithm="tpe", 
                                            early_stopping="asha", return_train_score=True, n_iter=num_iter, fold=fold,
                                            verbose=False, choose_better=False, custom_grid=custom_grid)
        results = self.pycaret.pull(pop=True)
        tuned_results = results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
        params = self.get_params(name, tuned_model)
        return tuned_model, tuned_results, params
    
    def stack_models(self, estimator_list: list[Any], fold: int=5, 
                     meta_model: Any=None) -> tuple[Any, pd.DataFrame, pd.Series]:
        """
        Create a stacked ensemble model from a list of models.

        Parameters
        ----------
        estimator_list : list[Any]
            A list of trained models to use for the ensemble.
        fold : int, optional
            The number of cross-validation folds to use. Defaults to 5.
        meta_model : Any, optional
            The meta model to use for stacking. Defaults to None.

        Returns
        -------
        tuple[Any, pd.DataFrame, pd.Series]
            A tuple containing the stacked ensemble model, the results of the ensemble, and the parameters used for training the ensemble.
        """
        self.log.info("----------Stacking the best models--------------")
        stacked_models = self.pycaret.stack_models(estimator_list, optimize=self.optimize, 
                                                 return_train_score=True,  verbose=False, fold=fold, 
                                                 meta_model_fold=fold, 
                                                 meta_model=meta_model)
        results = self.pycaret.pull(pop=True)
        stacked_results = results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
        params = self.get_params_stacked(stacked_models)
        return stacked_models, stacked_results, params
    
    def create_majority(self, estimator_list: list[Any], fold: int=5, 
                        weights: list[float] | None=None) -> tuple[Any, pd.DataFrame]:
        """
        Create a majority vote ensemble model from a list of models.

        Parameters
        ----------
        estimator_list : list[Any]
            A list of trained models to use for the ensemble.
        fold : int, optional
            The number of cross-validation folds to use. Defaults to 5.
        weights : list[float] or None, optional
            The weights to use for each model in the ensemble. Defaults to None.

        Returns
        -------
        tuple[Any, pd.DataFrame]
            A tuple containing the majority vote ensemble model and the results of the ensemble.
        """
        self.log.info("----------Creating a majority voting model--------------")
        self.log.info(f"fold: {fold}")
        self.log.info(f"weights: {weights}")
        majority_model = self.pycaret.blend_models(estimator_list, optimize=self.optimize, 
                                                 verbose=False, return_train_score=True, fold=fold, 
                                                 weights=weights)
        
        results = self.pycaret.pull(pop=True)
        majority_results = results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
        return majority_model, majority_results
    
    def finalize_model(self, model: Any):
        """
        Finalize the model by training it with all the data including test set

        Parameters
        ----------
        model : Any
            The model to finalize

        Returns
        -------
        Any
            The finalized model
        """
        self.log.info("----------Finalizing the model by training it with all the data including test set--------------")
        finalized = self.pycaret.finalize_model(model)
        return finalized
    
    def evaluate_model(self, model: Any, save: bool | str =False) -> None:
        """
        Evaluate the model by plotting the learning curve

        Parameters
        ----------
        model : pycaret.classification.Classifier or pycaret.regression.Regressor
            The model to evaluate.
        save : bool | str, optional
            Save the plots, by default False but you can also indicate the path for the plot
        """
        if isinstance(save, str):
            Path(save).mkdir(parents=True, exist_ok=True)
        self.pycaret.plot_model(model, "learning", save=save) # type: ignore

    def predict(self, estimador: Any, target_data: pd.DataFrame|None=None) -> pd.DataFrame:
        """
        Predict with the new data or if not specified predict on the holdout data.

        Parameters
        ----------
        estimador : Any
            The trained model.
        target_data : pd.DataFrame, optional
            The data to predict, by default None

        Returns
        -------
        pd.DataFrame, pd.Series
            The predictions are incorporated into the target data dataframe
        
        """
        if self.objective == "classification":
            pred = self.pycaret.predict_model(estimador, data=target_data, 
                                        verbose=False, raw_score=True) # type: ignore
        else:
            pred = self.pycaret.predict_model(estimador, data=target_data, 
                                              verbose=False)
        if target_data is None:
            results = self.pycaret.pull(pop=True)
            return results
        return pred
    
    def save(self, model: Any, filename: str, model_only: bool=False):
        """
        Save the model

        Parameters
        ----------
        model : Any
            The trained model.
        filename : str
            The name of the file to save the model.
        """
        self.pycaret.save_model(model, filename, model_only=model_only)
    
    def load_model(self, filename: str):
        """
        Load the model

        Parameters
        ----------
        filename : str
            The name of the file to load the model.
        """
        self.pycaret.load_model(filename)

    def get_logs(self):
        """
        Get the logs from the training in a dataframe format.

        Returns
        -------
        pd.DataFrame
            The logs from the training, if logs=True.
        """
        return self.pycaret.get_logs()

class Trainer:
    def __init__(self, caret_interface: PycaretInterface, training_arguments: ModelArguments, 
                 num_splits: int=5, test_size: float = 0.2, num_iter: int=50, cross_validation: bool=True):
        
        """
        Initialize a Trainer object with the given parameters.

        Parameters
        ----------
        caret_interface : PycaretInterface
            The model to use for training.
        training_arguments : ModelArguments
            The arguments to use for training, classification or regression.
        num_splits : int, optional
            The number of splits to use in cross-validation. Defaults to 5.
        test_size : float, optional
            The proportion of the data to use as a test set. Defaults to 0.2.
        num_iter : int, optional
            The number of iterations to use for tuning. Defaults to 30.
        cross_validation : bool, optional
            If True, use cross-validation. Defaults to True.
        """
        self.log = Log("model_training")
        self.log.info("----------------Trainer inputs-------------------------")
        self.num_splits = num_splits
        self.experiment = caret_interface
        self.num_iter = num_iter
        self.test_size = test_size
        self.log.info(f"Number of kfolds: {self.num_splits}")
        self.log.info(f"Number of retuning iterations: {self.num_iter}")
        self.log.info(f"Test size: {self.test_size}")
        self.arguments = training_arguments
        self.experiment.plots = self.arguments.plot
        self.cross_validation = cross_validation

    def rank_results(self, results: dict[str, pd.DataFrame], returned_models:dict[str, Any], 
                     scoring_function: Callable):
        """
        Rank the results based on the scores obtained from the scoring function.

        Parameters
        ----------
        results : dict
            A dictionary containing the results obtained from the models.
        returned_models : dict
            A dictionary containing the models that returned the results.
        scoring_function : Callable
            A function that takes in the results and returns a score.

        Returns
        -------
        pandas.DataFrame, dict
            A concatenated dataframe of the sorted results and a dictionary of the sorted models.

        Examples
        --------
        >>> results = {'model1': DataFrame[1, 2, 3], 'model2': DataFrame[4, 5, 6], 'model3': DataFrame[7, 8, 9]}
        >>> returned_models = {'model1': 'Model A', 'model2': 'Model B', 'model3': 'Model C'}
        >>> def scoring_function(x):
        ...     return sum(x)
        >>> rank_results(results, returned_models, scoring_function)

        The order would be like so
        model3  7  8  9
        model2  4  5  6
        model1  1  2  3, 
        {'model3': 'Model C', 'model2': 'Model B', 'model1': 'Model A'})
        """
        scores = {}
        for key, value in results.items():
            score_dataframe = scoring_function(value)
            scores[key] = score_dataframe

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        #sorted_scores = {key: scores[key] for key in sorted_keys}
        sorted_results = {key: results[key] for key in sorted_keys}
        sorted_models = {key: returned_models[key] for key in sorted_keys}

        return pd.concat(sorted_results), sorted_models
    
    def train(self, features: pd.DataFrame, label_name:str,
              **kwargs: Any) -> tuple[dict, dict]:
        """
        Train the models on the specified feature data and return the results and models.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            The training feature data.
        label_name : str
            The name of the label column.
        **kwargs :
            Additional parameters to pass to the setup_training function which calls the pycaret.setup function.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing the results and models.
        """
        # To access the transformed data
        self.experiment.setup_training(features, label_name, self.num_splits, self.test_size, **kwargs) # type: ignore
        if self.arguments.drop:
            self.experiment.final_models = [x for x in self.experiment.final_models if x not in self.arguments.drop]   
        if self.arguments.selected:
            self.experiment.final_models = self.arguments.selected
        if self.arguments.add:
            self.experiment.final_models += self.arguments.add

        results, returned_models = self.experiment.train(self.cross_validation)
        
        return results, returned_models
    
    def retune_best_models(self, sorted_models:dict[str, Any], custom_grid=None):
        """
        Retune the best models using the specified optimization metric and number of iterations.

        Parameters
        ----------
        sorted_models : dict[str, Any]
            A dictionary of sorted models.

        Returns
        -------
        Tuple[pd.DataFrame, dict, pd.DataFrame]
            A tuple containing the retuned model results, the retuned models, and the parameters used for retuning.
        """
        new_models = {}
        new_results = {}
        new_params = {}
        self.log.info("--------Retuning the best models--------")
        for key, model in list(sorted_models.items())[:self.experiment.best_model]:
            self.log.info(f"Retuning {key}")
            if key in ["nb", "svm", "par", "mlp"] or (key not in self.experiment.final_models and not custom_grid):
                new_models[key] = model
            else:
                tuned_model, results, params =  self.experiment.retune_model(key, model, self.num_iter, fold=self.num_splits, custom_grid=custom_grid)
                new_models[key] = tuned_model
                new_results[key] = results
                new_params[key] = params
                
        return pd.concat(new_results), new_models, pd.concat(new_params)
    
    def stack_models(self, sorted_models: dict[str, Any], meta_model: Any=None):
        """
        Create a stacked ensemble model from a dict of models.

        Parameters
        ----------
        sorted_models : dict[str, Any]
            A dictionary of sorted models.
        meta_model : Any or None, optional
            The meta model to use for stacking. Defaults to None.

        Returns
        -------
        Tuple[dict, Model, dict]
            A tuple containing the stacked ensemble results, the stacked ensemble model, and the parameters used for training the stacked ensemble model.
        """
        self.log.info("--------Stacking the best models--------")
  
        stacked_models, stacked_results, params = self.experiment.stack_models(list(sorted_models.values())[:self.experiment.best_model], 
                                                                               fold=self.num_splits, meta_model=meta_model)
        
        return stacked_results, stacked_models, params
    
    def create_majority_model(self, sorted_models: dict[str, Any], 
                               weights: list[float] | None =None) -> tuple[pd.DataFrame, Any]:
        """
        Create a majority vote ensemble model from a dictionary of sorted models.

        Parameters
        ----------
        sorted_models : dict
            A dictionary of sorted models.
        weights : Iterable[float] or None, optional
            The weights to use for the models. Defaults to None.

        Returns
        -------
        tuple[pd.DataFrame, Any]
            A tuple containing the ensemble results and the ensemble model.
        """
        self.log.info("--------Creating an ensemble model--------")
        
        ensemble_model, ensemble_results = self.experiment.create_majority(list(sorted_models.values())[:self.experiment.best_model],
                                                                           fold=self.num_splits, weights=weights)
        
        return ensemble_results, ensemble_model
    
    def predict(self, sorted_models: dict | list | Any, target_data: pd.DataFrame|None=None) -> pd.DataFrame:
        """
        Generate predictions on the test set using the specified models.

        Parameters
        ----------
        sorted_models : dict or list
            The sorted models to use for prediction.
        target_data : pd.DataFrame, optional
            The data to predict on. Defaults to None. Which then will use the holdout data

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the predictions on the test set.
        """
        match sorted_models:
            case [*list_models]:
                final = []
                # it keeps the original sorted order
                for mod in list_models:
                    result = self.experiment.predict(mod, target_data)
                    result = result.set_index("Model")
                    final.append(result)
                return pd.concat(final)
            
            case {**dict_models}: # for single models
                final = []
                for name, model in list(dict_models.items())[:self.experiment.best_model]:
                    result = self.experiment.predict(model, target_data)
                    result = result.set_index("Model")
                    result.index = [f"{name}_{x}" for x in result.index]
                    final.append(result)
                return pd.concat(final)

            case mod: # for stacked or majority models
                result = self.experiment.predict(mod, target_data)
                result = result.set_index("Model")
                return result
            
    def run_training(self, feature: pd.DataFrame, label_name: str,
                      **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
            """
            A function that splits the data into training and test sets and then trains the models
            using cross-validation but only on the training data

            Parameters
            ----------
            feature : pd.DataFrame
                The features of the training set
            label_name : str
                The column name of the label in the feature DataFrame

            **kwargs : dict, optional
                A dictionary containing the parameters for the setup function in pycaret.
            Returns
            -------
            pd.DataFrame
                A dictionary with the sorted results from pycaret
            list[models]
                A dictionary with the sorted models from pycaret
            pd.DataFrame
            
            """
            results, returned_models = self.train(feature, label_name, **kwargs) # type: ignore
            sorted_results, sorted_models = self.rank_results(results, returned_models, self.arguments._calculate_score_dataframe)
            top_params = self.experiment.get_best_params_multiple(sorted_models)

            return sorted_results, sorted_models, top_params
    
    def generate_training_results(self, feature: pd.DataFrame, label_name: str, tune: bool=False, 
                              **kwargs: Any) -> tuple[dict[str, dict], dict[str, dict]]:
        """
        Generate training results for a given model, training object, and feature data.

        Parameters
        ----------
        feature : pd.DataFrame
            The feature data to use.
        label_name : str
            The name of the label to use for training in the feature data.
        tune : bool, optional
            Whether to tune the hyperparameters. Defaults to True.
        **kwargs : dict
            Additional keyword arguments to pass to the pycaret setup function.

        Returns
        -------
        tuple[dict, dict]
            A tuple of dictionary containing the training results and the models.
        """    
        sorted_results, sorted_models, top_params = self.run_training(feature, label_name, **kwargs)
        if 'dummy' in sorted_results.index.unique(0)[:self.experiment.best_model]:
            warnings.warn(f"Dummy model is in the top {list(sorted_results.index.unique(0)).index('dummy')} models")
            tune = False
        if len(sorted_models) == 1 and tune:
            warnings.warn("Only one model was trained, turning off tuning")
            tune = False

        self.experiment.plot_best_models(sorted_models, "not_tuned")
        results, models_dict = self.save_results_and_models(sorted_results, sorted_models, top_params, "not_tuned")

        if tune and self.cross_validation:
            sorted_result_tune, sorted_models_tune, top_params_tune = self.retune_best_models(sorted_models)
            results_tuned, models_dict_tuned = self.save_results_and_models(sorted_result_tune, sorted_models_tune, top_params_tune, "tuned")
            self.experiment.plot_best_models(sorted_models_tune, "tuned")
            results.update(results_tuned)
            models_dict.update(models_dict_tuned)

        return results, models_dict

    def save_results_and_models(self, sorted_results: pd.DataFrame, sorted_models: dict[str, Any], 
                                top_params: pd.DataFrame | pd.Series, key: str):
        """
        Save the results and models in a dictionary.

        Parameters
        ----------
        sorted_results : pd.DataFrame
            The sorted results from the training.
        sorted_models : dict[str, Any]
            The sorted models from the training.
        top_params : pd.DataFrame | pd.Series
            The parameters of the models.
        key : str
            The key to use for the results and models (tuned and notuned)

        Returns
        -------
        dict, dict
            A dictionary containing the results and a dictionary containing the models.
        """
        results = defaultdict(dict)
        models_dict = defaultdict(dict)

        results[key]["train"] = sorted_results, top_params
        models_dict[key]["train"] = sorted_models

        if self.cross_validation:
            stacked_results, stacked_models, stacked_params = self.stack_models(sorted_models)
            results[key]["stacked"] = stacked_results, stacked_params
            majority_results, majority_models = self.create_majority_model(sorted_models)
            results[key]["majority"] = majority_results,  
            models_dict[key]["stacked"] = stacked_models
            models_dict[key]["majority"] = majority_models

        return results, models_dict
    
    def generate_holdout_prediction(self, models_dict: dict[str, dict]) -> dict[str, pd.DataFrame]:
        
        """
        Generate test set predictions for a given set of models.

        Parameters
        ----------
        models_dict : dict[str, dict]
            A dictionary containing the models to use for each tuning status.

        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary containing the prediction results for each tuning status.
        """
        prediction_results = {}
        for tune_status, result_dict in models_dict.items():
            predictions = []
            for key, models in result_dict.items():
                # get the test set prediction results
                predictions.append(self.predict(models))
            prediction_results[tune_status] = self.arguments.sort_holdout_prediction(pd.concat(predictions))
        return prediction_results

    def iterate_multiple_features(self, iterator: Iterator, training_output: Path, split_strategy: Any=None, split_index: int=0,
                                  filter_sheet: str | None | list[str] = None, test_size: float = 0.2,
                                  **kwargs: Any) -> None:
    
        """
        Iterates over multiple input features and generates training results for each feature.

        Parameters
        ----------
        iterator : Iterator
            An iterator that yields a tuple of input features and sheet names.
        training_output : Path
            The path to the directory where the training results will be saved.
        split_strategy : Any, optional
            The split strategy to use for training. It has to be a custom split function compatible
            with scikit-learn. Defaults to None which will use the default split strategy from pycaret (stratifiedkfold or kfold)
        test_size : float, optional
            The proportion of the data to use as a test set. Defaults to 0.2.
        split_index : int, optional
            The index to use for the split. Defaults to 0.
        filter_sheet : str | None | list[str], optional
            The sheet name or names to filter, it could be substring or exact matches. Defaults to None.
        Returns
        -------
        None
        """

        performance_list = []
        for input_feature, label_name, sheet in iterator:
            if filter_sheet is not None and sheet in filter_sheet:
                continue
            if split_strategy is not None:
                X_train, X_test, _, _ = split_strategy.train_test_split(input_feature, input_feature[label_name], 
                                                                        test_size=test_size, 
                                                                        split_index=split_index)
                sorted_results, sorted_models, top_params = self.run_training(X_train, label_name, test_data=X_test, 
                                                            fold_strategy=split_strategy, **kwargs)
            else:
                sorted_results, sorted_models, top_params = self.run_training(input_feature, label_name,
                                                                              **kwargs)
            index = sorted_results.index.unique(0)[:self.experiment.best_model]
            score = 0
            for i in index:
                score += self.arguments._calculate_score_dataframe(sorted_results.loc[i])
            performance_list.append((sheet, sorted_results.loc[index], score))
        performance_list.sort(key=lambda x: x[2], reverse=True)
        for sheet, performance, score in performance_list:
            write_results(training_output, performance, sheet_name=sheet)