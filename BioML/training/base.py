from dataclasses import dataclass, field
import warnings 
import pandas as pd
from typing import Iterable, Callable, Any
import numpy as np
from pathlib import Path
import time
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from ..utilities import Log, scale
from sklearn.metrics import average_precision_score
from ..utilities import write_excel
from collections import defaultdict

def write_results(training_output: Path, sorted_results: pd.DataFrame, top_params: pd.Series | None = None, 
                  sheet_name: str|None=None) -> None:
    """
    Writes the training results and top hyperparameters to Excel files.

    Parameters
    ----------
    training_output : Path
        The path to the directory where the Excel files will be saved.
    sorted_results : pd.DataFrame
        A pandas DataFrame containing the sorted training results.
    top_params : pd.Series or None, optional
        A pandas Series containing the top hyperparameters. Defaults to None.
    sheet_name : str or None, optional
        The name of the sheet to write the results to. Defaults to None.

    Returns
    -------
    None
    """
    write_excel(training_output / "training_results.xlsx", sorted_results, sheet_name)
    if top_params is not None:
        write_excel(training_output / f"top_hyperparameters.xlsx", top_params, sheet_name)

@dataclass
class DataParser:
    """
    A class for parsing feature and label data.

    Parameters
    ----------
    features : pd.DataFrame or str or list or np.ndarray
        The feature data.
    label : pd.Series or str or Iterable[int|float] or None, optional
        The label data. Defaults to None.
    outliers : dict[str, tuple], optional
        A dictionary containing the indices of the outliers in the feature and label data. Defaults to an empty dictionary.
    scaler : str, optional
        The type of scaler to use for feature scaling. Defaults to "robust".
    sheets : str or int, optional
        The sheet name or index to read from an Excel file. Defaults to 0.

    Attributes
    ----------
    features : pd.DataFrame
        The feature data.
    label : pd.Series or None
        The label data.
    outliers : dict[str, tuple]
        A dictionary containing the indices of the outliers in the feature and label data.
    scaler : str
        The type of scaler to use for feature scaling.
    sheets : str or int
        The sheet name or index to read from an Excel file.

    Methods
    -------
    read_features(features)
        Reads the feature data from a file or returns the input data.
    read_labels(label)
        Reads the label data from a file or returns the input data.
    scale(X_train, X_test)
        Scales the feature data.
    process(transformed_x, test_x, y_train, y_test)
        Concatenates the feature and label data so that it can be used by pycaret.
    """
    features: pd.DataFrame | str | list | np.ndarray
    label: pd.Series | str | Iterable[int|float] | None = None
    outliers: dict[str, tuple] = field(default_factory=lambda: defaultdict(tuple))
    scaler: str="robust"
    sheets: str | int = 0

    def __post_init__(self):
        self.features = self.read_features(self.features)
        if self.label:
            self.label = self.read_labels(self.label)

    def read_features(self, features: str | pd.DataFrame | list | np.ndarray) -> pd.DataFrame:
        """
        Reads the feature data from a file or returns the input data.

        Parameters
        ----------
        features : str or pd.DataFrame or list or np.ndarray
            The feature data.

        Returns
        -------
        pd.DataFrame
            The feature data as a pandas DataFrame.

        Raises
        ------
        TypeError
            If the input data type is not supported.
        """
        # concatenate features and labels
        if isinstance(features, str):
            if features.endswith(".csv"):
                return pd.read_csv(f"{features}", index_col=0) # the first column should contain the sample names

            elif features.endswith(".xlsx"):
                with pd.ExcelFile(features) as file:
                    if len(file.sheet_names) > 1:
                        warnings.warn(f"The excel file contains more than one sheet, only the sheet {self.sheets} will be used")

                return pd.read_excel(features, index_col=0, engine='openpyxl', sheet_name=self.sheets)
        
            
        elif isinstance(features, pd.DataFrame):
            return features
        elif isinstance(features, (list, np.ndarray)):
            return pd.DataFrame(features)

        self.log.error("features should be a csv or excel file, an array or a pandas DataFrame")
        raise TypeError("features should be a csv or excel file, an array or a pandas DataFrame")
        
    def read_labels(self, label: str | pd.Series) -> pd.Series:
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

        Raises
        ------
        TypeError
            If the input data type is not supported.
        """
        if isinstance(label, pd.Series):
            label.index.name = "target"
            return label
        
        elif isinstance(label, str):
            if Path(label).exists() and Path(label).suffix == ".csv":
                label = pd.read_csv(label, index_col=0)
                label.index.name = "target"
                return label    
            elif label in self.features.columns:
                lab = self.features[label]
                self.features.drop(label, axis=1, inplace=True)
                return lab
                    
        self.log.error("label should be a csv file, a pandas Series or inside features")
        raise TypeError("label should be a csv file, a pandas Series or inside features")
    
    def scale(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scales the train data and test data

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        X_test : pd.DataFrame
            The test data.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            The scaled training and test data.
        """
        #remove outliers
        X_train = X_train.loc[[x for x in X_train.index if x not in self.outliers["x_train"]]]
        X_test = X_test.loc[[x for x in X_test.index if x not in self.outliers["x_test"]]]

        transformed_x, scaler_dict, test_x = scale(self.scaler, X_train, X_test)

        return transformed_x, test_x
    
    def process(self, transformed_x: pd.DataFrame, test_x: pd.DataFrame, y_train: pd.Series, 
                y_test: pd.Series)-> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Concatenate the transformed feature data with the label data.

        Parameters
        ----------
        transformed_x : pd.DataFrame
            The transformed training feature data.
        test_x : pd.DataFrame
            The transformed testing feature data.
        y_train : pd.Series
            The training label data.
        y_test : pd.Series
            The testing label data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the concatenated training feature and label data, and the concatenated testing feature and label data.
        """
        transformed_x = pd.concat([transformed_x, y_train], axis=1) 
        test_x = pd.concat([test_x, y_test], axis=1)
        return transformed_x, test_x
        

@dataclass
class PycaretInterface:
    objective: str
    label_name: str
    seed: None | int = None
    #verbose: bool = False
    budget_time: None | int
    log = Log("model_training")
    best_model: int = 3
    output_path: Path | str | None= None 
    _plots: list[str] = field(init=False)
    model: ClassificationExperiment | RegressionExperiment = field(init=False)
    
    def __post_init__(self):
        if self.objective == "classification":
            self.model = ClassificationExperiment()
            self._plots = ["confusion_matrix", "learning", "class_report", "auc", "pr"]
        elif self.objective == "regression":
            self.model = RegressionExperiment()
            self._plots = ["residuals", "error", "learning"]
        if not self.seed:
            self.seed = int(time.time())
        self.log.info(f"Seed: {self.seed}")
        if isinstance(self.budget_time, (int, float)):
            if self.budget_time < 0:
                raise ValueError("The budget time should be greater than 0")
        elif self.budget_time is not None:
            raise ValueError("The budget time should be greater than 0 or None")
        if self.output_path is None:
            self.output_path = Path.cwd()
        else:
            self.output_path = Path(self.output_path)

        self.log.info("------------------------------------------------------------------------------")
        self.log.info("PycaretInterface parameters")
        self.log.info(f"Budget time: {self.budget_time}")
        self.log.info(f"Label name: {self.label_name}")
        self.log.info(f"The number of models to select: {self.best_model}")
        self.log.info(f"Output path: {self.output_path}")
    
    @staticmethod
    def _check_value(value, element_list, element_name):
        if isinstance(value, (list, np.ndarray, tuple, set)):
            test = list(set(value).difference(element_list))
            if test:
                raise ValueError(f"the {element_name} should be one of the following: {element_list} and not {test}")
        elif isinstance(value, str):
            if value not in element_list:
                raise ValueError(f"the {element_name} should be one of the following: {element_list} and not {value}")
            value = [value]
        else:
            raise TypeError(f"the {element_name} should be a string or an array of strings")
        return value

    @property
    def plots(self):
        """
        The plots that should be saved
        """
        return self._plots
    
    @plots.setter
    def plots(self, value) -> list[str]:
        self._plots = self._check_value(value, self._plots, "plots")

    @property
    def final_models(self):
        """
        The models to be used for classification or regression use one of the keys
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
        mod = self.model.models()
        self._final_models = mod.index.to_list()
        return self._final_models
    
    @final_models.setter
    def final_models(self, value) -> list[str]:
        self._final_models = self._check_value(value, self.mod.index.to_list(), "models")

    def setup_training(self, X_train, X_test, fold):
        if self.objective == "classification":
            self.model.setup(data=X_train, target=self.label_name, normalize=False, preprocess=False, 
                         log_experiment=False, experiment_name="Classification", 
                        session_id = self.seed, fold_shuffle=True, fold=fold, test_data=X_test)
        
            self.model.add_metric("averagePre", "Average Precision Score", average_precision_score, 
                                   average="weighted", target="pred_proba", multiclass=False)

        elif self.objective == "regression":
            self.model.setup(data=X_train, target=self.label_name, normalize=False, preprocess=False, 
                             log_experiment=False, experiment_name="Regression", 
                             session_id = self.seed, fold_shuffle=True,
                             fold=fold, test_data=X_test)

        return self.model

    def train(self):
        self.log.info("--------------------------------------------------------")
        self.log.info(f"Training {self.objective} models")
        self.log.info(f"The models used {self.final_models}")
        results = {}
        returned_models = {}
        np.random.seed(self.seed)
        np.random.shuffle(self.final_models)
        if self.budget_time:
            self.log.info(f"Time budget is {self.budget_time} minutes")
        runtime_start = time.time()
        for m in self.final_models:
            model =self.model.create_model(m, return_train_score=True, verbose=False)
            model_results = self.model.pull(pop=True)
            model_results = model_results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
            returned_models[m] = model
            results[m] = model_results
            runtime_train = time.time()
            total_runtime = (runtime_train - runtime_start) / 60
            
            if self.budget_time and total_runtime > self.budget_time:
                self.log.info(
                    f"Total runtime {total_runtime} is over time budget by {total_runtime - self.budget_time} min, breaking loop"
                )
                break
           
        self.log.info(f"Traininf over: Total runtime {total_runtime} min")

        return results, returned_models
    
    def plot_best_models(self, sorted_models, split_ind=None):
        
        self.log.info("Analyse the best models and plotting them")
        for ind, (name, model) in enumerate(sorted_models.items(), 1):
            if ind <= self.best_model:
                self.log.info(f"Analyse the top {ind} model: {name}")
                if not split_ind:
                    plot_path = self.output_path / "model_plots" / name
                else:
                    plot_path = self.output_path / "model_plots" / f"{name}" / f"split_{split_ind}"
                plot_path.mkdir(parents=True, exist_ok=True)
                for pl in self.plots:
                    self.model.plot_model(model, pl, save=plot_path)

    def get_params(self, name, model):
        if name != "catboost":
            params = model.get_params()
        else:
            params = model.get_all_params()
        params = {key: value for key, value in params.items() if key not in 
                    ["warm_start", "verbose", "oob_score"]}
        params = pd.Series(params)
        return params
    
    def get_params_stacked(stacked_models):
        params_dict = stacked_models.get_params()
        return pd.Series(params_dict["final_estimator"].get_params())

    def get_best_params_multiple(self, sorted_models: dict):
        model_params = {}
        for ind, (name, model) in enumerate(sorted_models.items(), 1):
            if ind > self.best_model:
                break
            params = self.get_params(name, model)
            model_params[name] = params

        return pd.concat(model_params)
    
    def retune_model(self, name, model, optimize="MCC", num_iter=5, fold=5):
        self.log.info("---------Retuning the best models--------------")
        self.log.info(f"optimize: {optimize}")
        self.log.info(f"num_iter: {num_iter}")
        self.log.info(f"fold: {fold}")
        tuned_model = self.model.tune_model(model, optimize=optimize, search_library="optuna", search_algorithm="tpe", 
                                            early_stopping="asha", return_train_score=True, n_iter=num_iter, fold=fold,
                                            verbose=False)
        results = self.model.pull(pop=True)
        tuned_results = results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
        params = self.get_params(name, tuned_model)
        return tuned_model, tuned_results, params
    
    def stack_models(self, estimator_list: list, optimize="MCC", fold=5, meta_model=None):
        self.log.info("----------Stacking the best models--------------")
        self.log.info(f"fold: {fold}")
        self.log.info(f"optimize: {optimize}")
        stacked_models = self.model.stack_models(estimator_list, optimize=optimize, 
                                                 return_train_score=True,  verbose=False, fold=fold, 
                                                 meta_model_fold=fold, 
                                                 meta_model=meta_model)
        results = self.model.pull(pop=True)
        stacked_results = results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
        params = self.get_params_stacked(stacked_models)
        return stacked_models, stacked_results, params
    
    def create_majority(self, estimator_list: list, optimize="MCC", fold=5, weights=None):
        self.log.info("----------Creating a majority voting model--------------")
        self.log.info(f"fold: {fold}")
        self.log.info(f"optimize: {optimize}")
        self.log.info(f"weights: {weights}")
        majority_model = self.model.blend_models(estimator_list, optimize=optimize, 
                                                 verbose=False, return_train_score=True, fold=fold, 
                                                 weights=weights)
        
        results = self.model.pull(pop=True)
        majority_results = results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
        return majority_model, majority_results
    
    def check_drift(self, input_data, target_data, filename=None):
        self.log.info("----------Checking for data drift--------------")
        self.model.check_drift(input_data, target_data, filename=filename)
        return filename
    
    def finalize_model(self, model):
        self.log.info("----------Finalizing the model by training it with all the data including test set--------------")
        finalized = self.model.finalize_model(model)
        return finalized
    
    def evaluate_model(self, model, save: bool | str =False):
        """
        Evaluate the model by plotting the learning curve

        Parameters
        ----------
        model : pycaret.classification.Classifier or pycaret.regression.Regressor
            The model to evaluate.
        save : bool | str, optional
            Save the plots, by default False
        """
        self.model.plot_model(model, "learning", save=save)

    def predict(self, estimador, target_data: pd.DataFrame|None=None) -> pd.DataFrame:
        """
        Predict with teh new data or if not specified predict on the holdout data.

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
            pred = self.model.predict_model(estimador, data=target_data, 
                                        verbose=False, raw_score=True)
        else:
            pred = self.model.predict_model(estimador, data=target_data, 
                                            verbose=False)
        if target_data is None or self.label_name in target_data.columns:
            results = self.model.pull(pop=True)
            return results
        return pred
    
    def save(self, model, filename: str):
        """
        Save the model

        Parameters
        ----------
        model : Any
            The trained model.
        filename : str
            The name of the file to save the model.
        """
        self.model.save_model(model, filename)
    
    def load_model(self, filename: str):
        """
        Load the model

        Parameters
        ----------
        filename : str
            The name of the file to load the model.
        """
        self.model.load_model(filename)


class Trainer:
    def __init__(self, model: PycaretInterface, num_splits: int=5):
        
        """
        Initialize a Trainer object with the given parameters.

        Parameters
        ----------
        model : PycaretInterface
            The model to use for training.
        num_splits : int, optional
            The number of splits to use in cross-validation. Defaults to 5.
        """
        self.log = Log("model_training")
        self.log.info("Reading the features")
        self.num_splits = num_splits
        self.experiment = model
        self.log.info(f"Number of kfolds: {self.num_splits}")

    def rank_results(self, results: dict[str, pd.DataFrame], returned_models:dict[str], 
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
        >>> results = {'model1': [1, 2, 3], 'model2': [4, 5, 6], 'model3': [7, 8, 9]}
        >>> returned_models = {'model1': 'Model A', 'model2': 'Model B', 'model3': 'Model C'}
        >>> def scoring_function(x):
        ...     return sum(x)
        >>> rank_results(results, returned_models, scoring_function)
        (   0  1  2
        model3  7  8  9
        model2  4  5  6
        model1  1  2  3, {'model3': 'Model C', 'model2': 'Model B', 'model1': 'Model A'})
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
    
    def train(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
              drop: tuple[str, ...] | None=None, selected_models: str | tuple[str, ...] | None=None):

        # To access the transformed data
        self.experiment.setup_training(X_train, X_test, self.num_splits)
        if drop:
            self.experiment.final_models = [x for x in self.experiment.final_models if x not in drop]   
        if selected_models:
            self.experiment.final_models = selected_models
        results, returned_models = self.experiment.train()
        
        return results, returned_models
    
    def analyse_models(self, transformed_x: pd.DataFrame, test_x: pd.DataFrame, scoring_fn: Callable, drop: tuple | None=None, 
                       selected: tuple | None=None) -> tuple[pd.DataFrame, dict, pd.Series]:
        """
        Analyze the trained models and rank them based on the specified scoring function.

        Parameters
        ----------
        transformed_x : pd.DataFrame
            The transformed feature data.
        test_x : pd.DataFrame
            The test feature data.
        scoring_fn : Callable
            The scoring function to use for evaluating the models.
        drop : tuple or None, optional
            The features to drop from the feature data. Defaults to None.
        selected : tuple or None, optional
            The features to select from the feature data. Defaults to None.

        Returns
        -------
        tuple[pd.DataFrame, dict, pd.Series]
            A tuple containing the sorted results and sorted models.
        """
        results, returned_models = self.train(transformed_x, test_x, drop, selected)
        sorted_results, sorted_models = self.rank_results(results, returned_models, scoring_fn)
        top_params = self.experiment.get_best_params_multiple(sorted_models)

        return sorted_results, sorted_models, top_params
    
    def retune_best_models(self, sorted_models:dict[str, Any], optimize: str="MCC", num_iter: int=5):
        """
        Retune the best models using the specified optimization metric and number of iterations.

        Parameters
        ----------
        sorted_models : dict[str, Any]
            A dictionary of sorted models.
        optimize : str, optional
            The metric to optimize for. Defaults to "MCC".
        num_iter : int, optional
            The number of iterations to use for retuning. Defaults to 5.

        Returns
        -------
        Tuple[pd.DataFrame, dict, pd.DataFrame]
            A tuple containing the retuned model results, the retuned models, and the parameters used for retuning.
        """
        new_models = {}
        new_results = {}
        new_params = {}
        self.log.info("--------Retuning the best models--------")
        for key, model in list(sorted_models.item())[:self.experiment.best_model]:
            self.log.info(f"Retuning {key}")
            tuned_model, results, params =  self.experiment.retune_model(key, model, optimize, num_iter, fold=self.num_splits)
            new_models[key] = tuned_model
            new_results[key] = results
            new_params[key] = params

        return pd.concat(new_results), new_models, pd.concat(new_params)
    
    def stack_models(self, sorted_models: dict[str, Any], optimize="MCC", meta_model: Any=None):
        """
        Create a stacked ensemble model from a dict of models.

        Parameters
        ----------
        sorted_models : dict[str, Any]
            A dictionary of sorted models.
        optimize : str, optional
            The metric to optimize for. Defaults to "MCC".
        meta_model : Any or None, optional
            The meta model to use for stacking. Defaults to None.

        Returns
        -------
        Tuple[dict, Model, dict]
            A tuple containing the stacked ensemble results, the stacked ensemble model, and the parameters used for training the stacked ensemble model.
        """
        self.log.info("--------Stacking the best models--------")
  
        stacked_models, stacked_results, params = self.experiment.stack_models(list(sorted_models.values())[:self.experiment.best_model], optimize=optimize, 
                                                                               fold=self.num_splits, meta_model=meta_model)
        
        return stacked_results, stacked_models, params
    
    def create_majority_model(self, sorted_models: dict[str, Any], optimize: str="MCC", 
                               weights: list[float] | None =None) -> tuple[pd.DataFrame, Any]:
        """
        Create a majority vote ensemble model from a dictionary of sorted models.

        Parameters
        ----------
        sorted_models : dict
            A dictionary of sorted models.
        optimize : str, optional
            The metric to optimize for. Defaults to "MCC".
        weights : Iterable[float] or None, optional
            The weights to use for the models. Defaults to None.

        Returns
        -------
        tuple[pd.DataFrame, Any]
            A tuple containing the ensemble results and the ensemble model.
        """
        self.log.info("--------Creating an ensemble model--------")
        
        ensemble_model, ensemble_results = self.experiment.create_majority(list(sorted_models.values())[:self.experiment.best_model], optimize=optimize,
                                                                           fold=self.num_splits, weights=weights)
        
        return ensemble_results, ensemble_model
    
    def predict_on_test_set(self, sorted_models: dict | list, name: str) -> pd.DataFrame:
        """
        Generate predictions on the test set using the specified models.

        Parameters
        ----------
        sorted_models : dict or list
            The sorted models to use for prediction.
        name : str
            The name of the test set.

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
                    result = self.experiment.predict(mod)
                    result.index = [f"Test-results-{name}"]
                    final.append(result)
                return pd.concat(final)
            
            case {**dict_models}: # for single models
                final = []
                for model in list(dict_models.values())[:self.experiment.best_model]:
                    result = self.experiment.predict(model)
                    result.index = [f"Test-results-{name}"]
                    final.append(result)
                return pd.concat(final)

            case mod: # for stacked or majority models
                result = self.experiment.predict(mod)
                result.index = [f"Test-results-{name}"]
                return result
           

def generate_training_results(model, training: Trainer, feature: DataParser, plot: tuple, optimize: str, 
                     tune: bool=True, strategy="holdout") -> dict[str, dict[str, tuple]]:
    """
    Generate training results for a given model, training object, and feature data.

    Parameters
    ----------
    model : Classifier or Regressor objects
        The model to use for training.
    training : Trainer
        The training object to use.
    feature : DataParser
        The feature data to use.
    plot : tuple
        A tuple containing the plot title and axis labels.
    optimize : str
        The metric to optimize for.
    tune : bool, optional
        Whether to tune the hyperparameters. Defaults to True.
    strategy : str, optional
        The strategy to use for training. Defaults to "holdout".

    Returns
    -------
    dict
        A dictionary containing the training results.
    """    
    sorted_results, sorted_models, top_params = model.run_training(training, feature, plot)

    # saving the results in a dictionary and writing it into excel files
    results = defaultdict(dict)
    results["not_tuned"][strategy] = sorted_results, sorted_models, top_params
    results["not_tuned"]["stacked"] = training.stack_models(sorted_models, optimize)
    results["not_tuned"]["majority"] = training.create_majority_model(sorted_models, optimize)

    if tune:
        sorted_result_tune, sorted_models_tune, top_params_tune = training.retune_best_models(sorted_models, optimize)
        results["tuned"][strategy] = sorted_result_tune, sorted_models_tune, top_params_tune
        results["tuned"]["stacked"] = training.stack_models(sorted_models_tune, optimize)
        results["tuned"]["majority"] = training.create_majority_model(sorted_models_tune, optimize)            
            

    return results