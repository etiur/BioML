from dataclasses import dataclass, field
import pandas as pd
from typing import Iterable, Callable
import numpy as np
from pathlib import Path
import time
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from ..utilities import Log, scale
from sklearn.metrics import average_precision_score
from ..utilities import write_excel
from collections import defaultdict

def write_results(training_output, sorted_results, top_params=None, sheet_name=None):
    write_excel(training_output / "training_results.xlsx", sorted_results, sheet_name)
    if top_params is not None:
        write_excel(training_output / f"top_hyperparameters.xlsx", top_params, sheet_name)

@dataclass
class DataParser:
    label: pd.Series | str | Iterable[int|float]
    features: pd.DataFrame | str | list | np.ndarray
    with_split: bool = field(init=False, default=False)

    def __post_init__(self):
        self.features = self.read_features(self.features)
        self.label = self.read_labels(self.label)
        if not isinstance(self.label, str):
            for key, value in self.features.items():
                self.features[key] = pd.concat([value, self.label], axis=1)
            self.label = "target"

    def read_features(self, features):
        # concatenate features and labels
        if isinstance(features, str):
            if features.endswith(".csv"):
                return {"model 1": pd.read_csv(f"{features}", index_col=0)} # the first column should contain the sample names

            elif features.endswith(".xlsx"):
                with pd.ExcelFile(features) as file:
                    sheet_names = file.sheet_names
                f = pd.read_excel(features, index_col=0, header=[0, 1], engine='openpyxl', sheet_name=sheet_names[0])
                if "split_0" in f.columns.unique(0):
                    self.with_split = True
                    del f
                    return pd.read_excel(features, index_col=0, header=[0, 1], engine='openpyxl', sheet_name=sheet_names)
        
                return pd.read_excel(features, index_col=0, engine='openpyxl', sheet_name=sheet_names)
                
        elif isinstance(features, pd.DataFrame):
            return {"model 1": features}
        elif isinstance(features, (list, np.ndarray)):
            return {"model 1": pd.DataFrame(features)}

        self.log.error("features should be a csv or excel file, an array or a pandas DataFrame")
        raise TypeError("features should be a csv or excel file, an array or a pandas DataFrame")
        
    
    def read_labels(self, label):
        if isinstance(label, pd.Series):
            label.index.name = "target"
            return label

        elif isinstance(label, str):
            if Path(label).exists() and Path(label).suffix == ".csv":
                label = pd.read_csv(label, index_col=0)
                label.index.name = "target"
                return label    
            elif label in list(self.features.values())[0].columns:
                return label
                    
        self.log.error("label should be a csv file, a pandas Series or inside features")
        raise TypeError("label should be a csv file, a pandas Series or inside features")
        

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
        pred = self.model.predict_model(estimador, data=target_data, 
                                        verbose=False, raw_score=True)
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
    def __init__(self, model: PycaretInterface, num_splits: int=5, 
                 test_size: float=0.2, outliers: tuple[str, ...]=(), scaler: str="robust"):
        
        """
        Initialize a Trainer object with the given parameters.

        Parameters
        ----------
        model : PycaretInterface
            The model to use for training.
        training_output : str, optional
            The path to the directory where the training results will be saved. Defaults to "training_results".
        num_splits : int, optional
            The number of splits to use in cross-validation. Defaults to 5.
        test_size : float, optional
            The proportion of the dataset to include in the test split. Defaults to 0.2.
        outliers : tuple[str, ...], optional
            The list of outliers to remove from the training and test sets. Defaults to ().
        scaler : str, optional
            The type of scaler to use for feature scaling. Defaults to "robust".
        """
        self.log = Log("model_training")
        self.log.info("Reading the features")
        self.num_splits = num_splits
        self.test_size = test_size
        self.scaler = scaler
        self.outliers = outliers
        self.experiment = model
        self.log.info(f"Test_size: {test_size}")
        self.log.info(f"Outliers: {', '.join(self.outliers)}")
        self.log.info(f"Scaler: {scaler}")
        self.log.info(f"Number of kfolds: {self.num_splits}")
    
    def return_train_test(self, train_index: pd.DataFrame | np.ndarray, test_index: pd.DataFrame | np.ndarray, 
                          features: pd.DataFrame | np.ndarray = None, strategy: str="holdout", split_index: int | None=None):
        # split and filter
        if self.with_split:
            feat_subset = features.loc[:, f"split_{split_index}"]  # each split different features and different fold of
            # training and test data
        else:
            feat_subset = features

        if strategy == "kfold":
            X_train = feat_subset.iloc[train_index]
            X_test = feat_subset.iloc[test_index]
        elif strategy == "holdout":
            X_train = train_index
            X_test = test_index

        #remove outliers
        X_train = X_train.loc[[x for x in X_train.index if x not in self.outliers]]
        X_test = X_test.loc[[x for x in X_test.index if x not in self.outliers]]

        transformed_x, scaler_dict, test_x = scale(self.scaler, X_train, X_test)

        return transformed_x, test_x

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
    
    def train(self, X_train, X_test, drop: tuple[str, ...] | None=None, selected_models: str | tuple[str, ...] | None=None):

        # To access the transformed data
        self.experiment.setup_training(X_train, X_test, self.num_splits)
        if drop:
            self.experiment.final_models = [x for x in self.experiment.final_models if x not in drop]   
        if selected_models:
            self.experiment.final_models = selected_models
        results, returned_models = self.experiment.train()
        
        return results, returned_models
    
    def analyse_models(self, transformed_x, test_x, scoring_fn, drop=None, selected=None):
        results, returned_models = self.train(transformed_x, test_x, drop, selected)
        sorted_results, sorted_models = self.rank_results(results, returned_models, scoring_fn)
        top_params = self.experiment.get_best_params_multiple(sorted_models)

        return sorted_results, sorted_models, top_params

    def _scale(self, train_index, test_index, features = None, strategy="holdout", split_index=None, 
               with_split=False):
        # split and filter
        if with_split:
            feat_subset = features.loc[:, f"split_{split_index}"]  # each split different features and different fold of
            # training and test data
        else:
            feat_subset = features

        if strategy == "kfold":
            X_train = feat_subset.iloc[train_index]
            X_test = feat_subset.iloc[test_index]
        elif strategy == "holdout":
            X_train = train_index
            X_test = test_index

        #remove outliers
        X_train = X_train.loc[[x for x in X_train.index if x not in self.outliers]]
        X_test = X_test.loc[[x for x in X_test.index if x not in self.outliers]]

        transformed_x, scaler_dict, test_x = scale(self.scaler, X_train, X_test)

        return transformed_x, test_x
    
    def setup_training(self, X_train, X_test, scoring_fn: Callable, plot: tuple=(), drop: tuple|None=None, selected: tuple | None=None):
        transformed_x, test_x = self._scale(X_train, X_test)
        sorted_results, sorted_models, top_params = self.analyse_models(transformed_x, test_x, scoring_fn, drop, selected)
        if plot:
            self.experiment.plots = plot
            self.experiment.plot_best_models(sorted_models)
        return sorted_results, sorted_models, top_params
    
    def retune_best_models(self, sorted_models:dict, optimize: str="MCC", num_iter: int=5):
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
    
    def stack_models(self, sorted_models: dict, optimize="MCC", meta_model=None):
        self.log.info("--------Stacking the best models--------")
  
        stacked_models, stacked_results, params = self.experiment.stack_models(list(sorted_models.values())[:self.experiment.best_model], optimize=optimize, fold=self.num_splits, 
                                                      meta_model=meta_model)
        
        return stacked_results, stacked_models, params
    
    def create_majority_model(self, sorted_models: dict, optimize: str="MCC", 
                               weights: Iterable[float] | None =None):
        self.log.info("--------Creating an ensemble model--------")
        
        ensemble_model, ensemble_results = self.experiment.create_majority(list(sorted_models.values())[:self.experiment.best_model], optimize=optimize, fold=self.num_splits, 
                                                        weights=weights)
        
        return ensemble_results, ensemble_model
    
    def predict_on_test_set(self, sorted_models: dict | list, name: str) -> pd.DataFrame:
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
                     tune: bool=True, strategy="holdout"):
        
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