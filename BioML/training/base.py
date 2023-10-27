from dataclasses import dataclass, field
import pandas as pd
from typing import Callable, Any
import numpy as np
from pathlib import Path
import time
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from ..utilities import Log
from sklearn.metrics import average_precision_score
from typing import Iterable
        

@dataclass
class PycaretInterface:
    """
    A class for interfacing with the PyCaret library for machine learning. 
    Learn more here:
    https://pycaret.gitbook.io/docs/
    https://medium.com/analytics-vidhya/pycaret-101-for-beginners-27d9aefd34c5

    Attributes
    ----------
    objective : str
        The objective of the machine learning model, either "classification" or "regression".
    label_name : str
        The name of the target variable in the dataset.
    optimize : str, optional
        The metric to optimize for. Defaults to "MCC".
    seed : None or int, optional
        The random seed to use for reproducibility. Defaults to None.
    budget_time : None or int, optional
        The time budget for training the models in minutes. Defaults to None.
    log : Log
        The logger for the PycaretInterface class.
    best_model : int, optional
        The number of best models to select. Defaults to 3.
    output_path : Path or str or None, optional
        The path to save the output files. Defaults to None.
    _plots : list of str
        The list of plots to generate for the models.
    model : ClassificationExperiment or RegressionExperiment
        The PyCaret experiment object for the machine learning model.

    Methods
    -------
    setup_training(X_train, X_test, fold)
        Set up the training data for the machine learning models.
    train()
        Train the machine learning models.
    get_best_models(results)
        Get the best models from the training results.
    evaluate_models(returned_models, X_test, y_test)
        Evaluate the performance of the trained models on the test data.
    save_models(returned_models)
        Save the trained models to files.
    load_models(model_names)
        Load the trained models from files.
    predict(X_test, model_names)
        Generate predictions on new data using the trained models.
    evaluate(X_test, y_test, model_names)
        Evaluate the performance of the trained models on new data.
    plot_best_models(sorted_models, split_ind=None)
        Analyze the best models and plot them.
    retune_model(name, model,, num_iter=5, fold=5)
        Retune the specified model using Optuna.
    stack_models(estimator_list, fold=5, meta_model=None)
        Create a stacked ensemble model from a list of models.
    get_params(name, model)
        Get the parameters of a trained model.
    get_best_params_multiple(sorted_models)
        Get the best parameters for multiple models.
    get_params_stacked(stacked_models)
        Get the parameters of the final estimator in a stacked ensemble model.

    """
    objective: str
    label_name: str
    seed: None | int = None
    optimize: str = "MCC"
    #verbose: bool = False
    budget_time: None | int = None
    log = Log("model_training")
    best_model: int = 3
    output_path: Path | str | None= None 
    _plots: list[str] = field(init=False)
    model: ClassificationExperiment | RegressionExperiment = field(init=False)
    
    def __post_init__(self):
        if self.objective == "classification":
            self.pycaret = ClassificationExperiment()
            self._plots = ["confusion_matrix", "learning", "class_report", "auc", "pr"]
            self._final_models = ('lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 
                                'mlp', 'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 
                                'lightgbm', 'catboost', 'dummy')

        elif self.objective == "regression":
            self.pycaret = RegressionExperiment()
            self._plots = ["residuals", "error", "learning"]
            self._final_models = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 
                                 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 
                                 'lightgbm', 'catboost', 'dummy']
        if not self.seed:
            self.seed = int(time.time())
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
        self.log.info(f"Seed: {self.seed}")
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
        return self._final_models
    
    @final_models.setter
    def final_models(self, value) -> list[str]:
        self._final_models = self._check_value(value, self._final_models, "models")

    def setup_training(self,X_train: pd.DataFrame, X_test: pd.DataFrame, fold: int) -> Any:
        """
        Call pycaret set_up for the training.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        X_test : pd.DataFrame
            The test data.
        fold : int
            The number of cross-validation folds to use.

        Returns
        -------
        Any
            The pycaret classification or regression object
        """
        if self.objective == "classification":
            self.pycaret.setup(data=X_train, target=self.label_name, normalize=False, preprocess=False, 
                         log_experiment=True, experiment_name="Classification", 
                        session_id = self.seed, fold_shuffle=True, fold=fold, test_data=X_test, verbose=False)
        
            self.pycaret.add_metric("averagePre", "Average Precision Score", average_precision_score, 
                                   average="weighted", target="pred_proba", multiclass=False)

        elif self.objective == "regression":
            self.pycaret.setup(data=X_train, target=self.label_name, normalize=False, preprocess=False, 
                             log_experiment=True, experiment_name="Regression", 
                             session_id = self.seed, fold_shuffle=True,
                             fold=fold, test_data=X_test, verbose=False)

        return self.pycaret

    def train(self):
        """
        Train the machine learning models.

        Returns
        -------
        Tuple[dict, dict]
            A tuple containing the results of the training and the trained models.
        """
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
            model =self.pycaret.create_model(m, return_train_score=True, verbose=False)
            model_results = self.pycaret.pull(pop=True)
            model_results = model_results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
            returned_models[m] = model
            results[m] = model_results
            runtime_train = time.time()
            total_runtime = round((runtime_train - runtime_start) / 60, 3)
            
            if self.budget_time and total_runtime > self.budget_time:
                self.log.info(
                    f"Total runtime {total_runtime} is over time budget by {total_runtime - self.budget_time} minutes, breaking loop"
                )
                break
           
        self.log.info(f"Traininf over: Total runtime {total_runtime} minutes")

        return results, returned_models
    
    def plot_best_models(self, sorted_models: dict[str, Any], split_ind: int | None=None):
        """
        Analyze the best models and plot them.

        Parameters
        ----------
        sorted_models : dict
            A dictionary containing the trained models sorted by performance.
        split_ind : int, optional
            The index of the data split to use for plotting. Defaults to None.
        """
        
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
    
    def get_params_stacked(self, stacked_models):
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
            params = self.get_params(name, model)
            model_params[name] = params

        return pd.concat(model_params)
    
    def retune_model(self, name: str, model: Any, num_iter: int=5, 
                     fold: int=5) -> tuple[Any, pd.DataFrame, pd.Series]:
        """
        Retune the specified model using Optuna.

        Parameters
        ----------
        name : str
            The name of the model to retune.
        model : Any
            The trained model to retune.
        num_iter : int, optional
            The number of iterations to use for tuning. Defaults to 5.
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
                                            verbose=False)
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
    
    def check_drift(self, input_data: pd.DataFrame, target_data: pd.DataFrame|None, 
                    filename: str | None=None) -> str | None:
        """
        Check for data drift

        Parameters
        ----------
        input_data : pd.DataFrame
            The input data.
        target_data : pd.DataFrame, optional
            The target data, by default None
        filename : str, optional
            Where the save the report in html format

        Returns
        -------
        str | None
            The file name
        """
        self.log.info("----------Checking for data drift--------------")
        self.pycaret.check_drift(input_data, target_data, filename=filename)
        return filename
    
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
        if not isinstance(save, bool):
            Path(save).mkdir(parents=True, exist_ok=True)
        self.pycaret.plot_model(model, "learning", save=save)

    def predict(self, estimador: Any, target_data: pd.DataFrame|None=None) -> pd.DataFrame:
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
            pred = self.pycaret.predict_model(estimador, data=target_data, 
                                        verbose=False, raw_score=True)
        else:
            pred = self.pycaret.predict_model(estimador, data=target_data, 
                                            verbose=False)
        if target_data is None or self.label_name in target_data.columns:
            results = self.pycaret.pull(pop=True)
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
        self.pycaret.save_model(model, filename)
    
    def load_model(self, filename: str):
        """
        Load the model

        Parameters
        ----------
        filename : str
            The name of the file to load the model.
        """
        self.pycaret.load_model(filename)


class Trainer:
    def __init__(self, caret_interface: PycaretInterface, num_splits: int=5):
        
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
        self.experiment = caret_interface
        self.log.info(f"Number of kfolds: {self.num_splits}")

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
        >>> results = {'model1': [1, 2, 3], 'model2': [4, 5, 6], 'model3': [7, 8, 9]}
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
    
    def train(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
              drop: tuple[str, ...] | None=None, selected_models: str | tuple[str, ...] | None=None) -> tuple[dict, dict]:
        """
        Train the models on the specified feature data and return the results and models.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        X_test : pd.DataFrame
            The test feature data.
        drop : tuple[str, ...] or None, optional
            The features to drop from the feature data. Defaults to None.
        selected_models : str or tuple[str, ...] or None, optional
            The models to use for training. Defaults to None.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing the results and models.
        """
        # To access the transformed data
        self.experiment.setup_training(X_train, X_test, self.num_splits)
        if drop:
            self.experiment.final_models = [x for x in self.experiment.final_models if x not in drop]   
        if selected_models:
            self.experiment.final_models = selected_models
        results, returned_models = self.experiment.train()
        
        return results, returned_models
    
    def analyse_models(self, transformed_x: pd.DataFrame, test_x: pd.DataFrame, scoring_fn: Callable, 
                       drop: tuple[str] | None=None, selected: tuple[str] | None=None) -> tuple[pd.DataFrame, dict, pd.Series]:
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
    
    def retune_best_models(self, sorted_models:dict[str, Any], num_iter: int=5):
        """
        Retune the best models using the specified optimization metric and number of iterations.

        Parameters
        ----------
        sorted_models : dict[str, Any]
            A dictionary of sorted models.
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
        for key, model in list(sorted_models.items())[:self.experiment.best_model]:
            self.log.info(f"Retuning {key}")
            tuned_model, results, params =  self.experiment.retune_model(key, model, num_iter, fold=self.num_splits)
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
                    result = result.set_index("Model")
                    final.append(result)
                return pd.concat(final)
            
            case {**dict_models}: # for single models
                final = []
                for model in list(dict_models.values())[:self.experiment.best_model]:
                    result = self.experiment.predict(model)
                    result = result.set_index("Model")
                    final.append(result)
                return pd.concat(final)

            case mod: # for stacked or majority models
                result = self.experiment.predict(mod)
                result = result.set_index("Model")
                return result
           

