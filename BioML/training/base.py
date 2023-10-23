from dataclasses import dataclass, field
from json import load
import stat
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
    _plots: list[str] = field(init=False)
    model: ClassificationExperiment | RegressionExperiment = field(init=False)

    def __post_init__(self):
        if self.objective == "classification":
            self.model = ClassificationExperiment()
            self._plots = ["confusion_matrix", "learning", "class_report", "auc", "pr"]
        elif self.objective == "regression":
            self.model = RegressionExperiment()
            self._plots = ["residuals", "error", "learning"]
        mod = self.models.models()
        self._final_models = mod.index.to_list()
        if not self.seed:
            self.seed = int(time.time())
        self.log.info(f"Seed: {self.seed}")
        if isinstance(self.budget_time, (int, float)):
            if self.budget_time < 0:
                raise ValueError("The budget time should be greater than 0")
        elif self.budget_time is not None:
            raise ValueError("The budget time should be greater than 0 or None")
        self.log.info("------------------------------------------------------------------------------")
        self.log.info("PycaretInterface parameters")
        self.log.info(f"Budget time: {self.budget_time}")
        self.log.info(f"Label name: {self.label_name}")
        self.log.info(f"The number of models to select: {self.best_model}")
    
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
        The models to be used for classification
        """
        return self._final_models
    
    @final_models.setter
    def final_models(self, value) -> list[str]:
        self._final_models = self._check_value(value, self.mod.index.to_list(), "models")

    def setup_training(self, X_train, X_test):
        if self.objective == "classification":
            self.model.setup(data=X_train, target=self.label_name, normalize=False, preprocess=False, 
                         log_experiment=False, experiment_name="Classification", 
                        session_id = self.seed, fold_shuffle=True, fold=10, test_data=X_test)
        
            self.model.add_metric("averagePre", "Average Precision Score", average_precision_score, 
                                   average="weighted", target="pred_proba", multiclass=False)

        elif self.objective == "regression":
            self.model.setup(data=X_train, target=self.label_name, normalize=False, preprocess=False, 
                             log_experiment=False, experiment_name="Regression", 
                             session_id = self.seed, fold_shuffle=True, 
                             fold=10, test_data=X_test)

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
    
    def plot_best_models(self, output_path, sorted_models, split_ind=None):
        
        self.log.info("Analyse the best models and plotting them")
        for ind, (name, model) in enumerate(sorted_models.items(), 1):
            if ind <= self.best_model:
                self.log.info(f"Analyse the top {ind} model: {name}")
                if not split_ind:
                    plot_path = output_path / "model_plots" / name
                else:
                    plot_path = output_path / "model_plots" / f"{name}" / f"split_{split_ind}"
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
    
    def stack_models(self, estimator_list: list, optimize="MCC", fold=5, probability_threshold: float | None=None, meta_model=None):
        self.log.info("----------Stacking the best models--------------")
        self.log.info(f"fold: {fold}")
        self.log.info(f"probability_threshold: {probability_threshold}")
        self.log.info(f"optimize: {optimize}")
        stacked_models = self.model.stack_models(estimator_list, optimize=optimize, 
                                                 return_train_score=True,  verbose=False, fold=fold, 
                                                 probability_threshold=probability_threshold, meta_model_fold=fold, 
                                                 meta_model=meta_model)
        results = self.model.pull(pop=True)
        stacked_results = results.loc[[("CV-Train", "Mean"), ("CV-Train", "Std"), ("CV-Val", "Mean"), ("CV-Val", "Std")]]
        params = self.get_params_stacked(stacked_models)
        return stacked_models, stacked_results, params
    
    def create_majority(self, estimator_list: list, optimize="MCC", fold=5, probability_threshold: float | None=None, weights=None):
        self.log.info("----------Creating a majority voting model--------------")
        self.log.info(f"fold: {fold}")
        self.log.info(f"probability_threshold: {probability_threshold}")
        self.log.info(f"optimize: {optimize}")
        self.log.info(f"weights: {weights}")
        majority_model = self.model.blend_models(estimator_list, optimize=optimize, 
                                                 verbose=False, return_train_score=True, fold=fold, 
                                                 probability_threshold=probability_threshold, weights=weights)
        
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
        Evaluate the mode by plotting the learning curve

        Parameters
        ----------
        model : pycaret.classification.Classifier or pycaret.regression.Regressor
            The model to evaluate.
        save : bool | str, optional
            Save the plots, by default False
        """
        self.model.plot_model(model, "learning", save=save)

    def predict(self, estimador, target_data: pd.DataFrame|None=None, 
                probability_threshold: float | None=None) -> pd.DataFrame:
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
                                        probability_threshold=probability_threshold, verbose=False)
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
    def __init__(self, model: PycaretInterface, output: str="training_results", num_splits: int=5, 
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
        self.output_path = Path(output)  # for the model results
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.experiment = model
        self.log.info(f"Test_size: {test_size}")
        self.log.info(f"Outliers: {', '.join(self.outliers)}")
        self.log.info(f"Scaler: {scaler}")
        self.log.info(f"Number of kfolds: {self.num_splits}")
        self.log.info(f"The output path: {self.output_path}")
    
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
    
    def train(self, X_train, X_test):

        # To access the transformed data
        self.experiment.setup_training(X_train, X_test)
        
        results, returned_models = self.experiment.train()
        
        return results, returned_models
    
    def analyse_models(self, transformed_x, test_x, scoring_fn):
        results, returned_models = self.train(transformed_x, test_x)
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
    
    def setup_kfold(self, feature, label_name, split_function, scoring_fn, plot=(), with_split=False):
        """
        A function that splits the data into kfolds of training and test sets and then trains the models
        using cross-validation but only on the training data. We are getting the performance 
        of using different hold-outs.

        Parameters
        ----------
        feature : pd.DataFrame
            A dataframe containing the training samples and the features
        plot : bool, optional
            Plot the plots relevant to the models, by default None
                1. learning: learning curve
                2. pr: Precision recall curve
                3. auc: the ROC curve
                4. confusion_matrix 
                5. class_report: read classification_report from sklearn.metrics

        Returns
        -------
        tuple(pd.DataFrame, dict[str, list[models]], pd.DataFrame)
            A tuple with the sorted results, sorted models from pycaret organized by split index or kfold index and the parameters
        """
        res = {}
        top_params = {}
        mod = {}

        for ind, (train_index, test_index) in enumerate(split_function.split(feature, feature[label_name])):
            transformed_x, test_x = self._scale(feature, train_index, test_index, strategy="kfold", split_index=ind,
                                                with_split=with_split)
            sorted_results, sorted_models, top_params = self.analyse_models(transformed_x, test_x, scoring_fn)

            res[f"split_{ind}"] = sorted_results
            top_params[f"split_{ind}"] = top_params
            mod[f"split_{ind}"] = sorted_models

            if plot:
                self.experiment.plots = plot
                self.experiment.plot_best_models(self.output_path, sorted_models, split_ind=ind)

        return pd.concat(res, axis=1), mod, pd.concat(top_params)
    
    def setup_holdout(self, X_train, X_test, scoring_fn, plot=()):
        transformed_x, test_x = self._scale(X_train, X_test)
        sorted_results, sorted_models, top_params = self.analyse_models(transformed_x, test_x, scoring_fn)
        if plot:
            self.experiment.plots = plot
            self.experiment.plot_best_models(sorted_models)
        return sorted_results, sorted_models, top_params
    
    def _retune_best_models(self, sorted_models:dict, optimize: str="MCC", num_iter: int=5):
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
    
    def _stack_models(self, sorted_models: dict, optimize="MCC",  probability_theshold: None|float=None, meta_model=None):
        self.log.info("--------Stacking the best models--------")
        if "split" in list(sorted_models)[0]:
            new_models = {}
            new_results = {}
            new_params = {}
            for key, sorted_model_by_split in sorted_models.items():
                new_models[key], new_results[key],
                new_params[key] = self.experiment.stack_models(list(sorted_model_by_split.values())[:self.experiment.best_model], 
                                                      optimize=optimize, fold=self.num_splits,
                                                      probability_threshold=probability_theshold, meta_model=meta_model)
                
            return pd.concat(new_results, axis=1), new_models, pd.concat(new_params)
        
        stacked_models, stacked_results, params = self.experiment.stack_models(list(sorted_models.values())[:self.experiment.best_model], optimize=optimize, fold=self.num_splits, 
                                                      probability_threshold=probability_theshold, meta_model=meta_model)
        
        return stacked_results, stacked_models, params
    
    def _create_majority_model(self, sorted_models: dict, optimize: str="MCC", probability_theshold: None|float=None, 
                               weights: Iterable[float] | None =None):
        self.log.info("--------Creating an ensemble model--------")
        if "split" in list(sorted_models)[0]:
            new_models = {}
            new_results = {}
            for key, sorted_model_by_split in sorted_models.items():
                new_models[key], new_results[key] = self.experiment.create_majority(list(sorted_model_by_split.values())[:self.experiment.best_model], optimize=optimize, fold=self.num_splits, 
                                                        probability_threshold=probability_theshold, weights=weights)
            return pd.concat(new_results, axis=1), new_models
        
        ensemble_model, ensemble_results = self.experiment.create_majority(list(sorted_models.values())[:self.experiment.best_model], optimize=optimize, fold=self.num_splits, 
                                                        probability_threshold=probability_theshold, weights=weights)
        
        return ensemble_results, ensemble_model
    
    def _predict_on_test_set(self, sorted_models: dict | list, name: str) -> pd.DataFrame:
        match sorted_models:
            case [*list_models]:
                final = []
                # it keeps the original sorted order
                for mod in list_models:
                    result = self.experiment.predict(mod)
                    result.index = [f"Test-results-{name}"]
                    final.append(result)
                return pd.concat(final)
            
            case {**dict_models} if "split" in list(dict_models)[0]: # for kfold models, kfold stacked or majority models:
                final = {}
                for split_ind, mod in dict_models.items():
                    if isinstance(mod, dict):
                        results_by_split = []
                        for model in list(mod.values())[:self.experiment.best_model]:
                            result = self.experiment.predict(model)
                            result.index = [f"Test-results-{name}"]
                            results_by_split.append(result)
                        final[f"split_{split_ind}"] = pd.concat(results_by_split)
                    else:
                        result = self.experiment.predict(mod)
                        result.index = [f"Test-results-{name}"]
                        final[f"split_{split_ind}"] = result

                return pd.concat(final)
            
            case {**dict_models}: # for single models
                final = []
                for model in list(dict_models.values())[:self.experiment.best_model]:
                    result = self.experiment.predict(model)
                    result.index = [f"Test-results-{name}"]
                    final.append(result)
                return pd.concat(final)

            case mod:
                result = self.experiment.predict(mod)
                result.index = [f"Test-results-{name}"]
                return result
           
    def _finalize_model(self, sorted_model, index: int | dict[str, int] | None = None):
        """
        Finalize the model by training it with all the data including the test set.

        Parameters
        ----------
        sorted_model : Any
            The model or models to finalize.

        Returns
        -------
        Any
            The finalized models

        Raises
        ------
        TypeError
            The model should be a list of models, a dictionary of models or a model
        """
        match sorted_model:
            case [*list_models]:
                final = []
                for mod in list_models:
                    final.append(self.experiment.finalize_model(mod))
                return final
            
            case {**dict_models} if "split" in list(dict_models)[0]: # for kfold models, kfold stacked or majority models
                final = {}
                for split_ind, value in dict_models.items():
                    if isinstance(value, dict):
                        model_name, mod = list(value.items())[index[split_ind]]
                        if split_ind not in final:
                            final[split_ind] = {}
                        final[split_ind][model_name] = self.experiment.finalize_model(mod)
                    else:
                        final[split_ind] = self.experiment.finalize_model(value)
                return final
            
            case {**dict_models}: # for holdout models
                final = {}
                model_name, mod = list(dict_models.items())[index]
                final[model_name] = self.experiment.finalize_model(mod)
                return final
            
            case model:
                final = self.experiment.finalize_model(model)
                return final
            
    def _save_model(self, sorted_models, filename: str | dict[str, int] | None=None, 
                    index:int | dict[str, int] |None=None):
        """
        Save the model

        Parameters
        ----------
        model : Any
            The trained model.
        filename : str, dict[str, str]
            The name of the file to save the model.
        """
        model_output = self.output_path / "models"
        match sorted_models:
            case {**models} if "split" in list(models)[0]: # for kfold models, kfold stacked or majority models 
                for split_ind, value in models.items():
                    if isinstance(value, dict):
                        model_name, mod = list(value.items())[index[split_ind]]
                        self.experiment.save(mod, model_output/str(split_ind)/model_name)
                    else:
                        self.experiment.save(value, model_output/filename[split_ind])
            case {**models}: #
                for model_name, mod in models.items():
                    self.experiment.save(mod, model_output/model_name)
            
            case other:
                self.experiment.save(other, model_output/filename)

            
            

