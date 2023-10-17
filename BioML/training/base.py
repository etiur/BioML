from dataclasses import dataclass, field
import pandas as pd
from typing import Iterable
import numpy as np
from pathlib import Path
import time
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from ..utilities import Log, scale
from sklearn.metrics import average_precision_score


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
    verbose: bool = False
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
        self.log.info(f"Best model: {self.best_model}")
    
    @property
    def plots(self):
        return self._plots
    
    @plots.setter
    def plots(self, value):
        if isinstance(value, (list, np.ndarray, tuple, set)):
            test = list(set(value).difference(self._plots))
            if test:
                raise ValueError(f"the plots should be one of the following: {self._plots} and not {test}")
        elif isinstance(value, str):
            if value not in self._plots:
                raise ValueError(f"the plots should be one of the following: {self._plots} and not {value}")
            value = [value]
        self._plots = value

    @property
    def final_models(self):
        """
        The models to be used for classification
        """
        return self._final_models
    
    @final_models.setter
    def final_models(self, value) -> list[str]:
        if isinstance(value, (list, np.ndarray, tuple, set)):
            test = list(set(value).difference(self.mod.index.to_list()))
            if test:
                raise ValueError(f"the models should be one of the following: {self.mod.index.to_list()} and not {test}")
        elif isinstance(value, str):
            if value not in self.mod.index.to_list():
                raise ValueError(f"the models should be one of the following: {self.mod.index.to_list()} and not {value}")
            value = [value]
        else:
            raise TypeError("the models should be a list or a string")
        
        self._final_models = value

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
            if not self.verbose:
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

    def get_params(self, sorted_models):
        model_params = {}
        for ind, (name, model) in enumerate(sorted_models.items(), 1):
            if ind > self.best_model:
                break
            if name != "catboost":
                params = model.get_params()
            else:
                params = model.get_all_params()
            params = {key: value for key, value in params.items() if key not in 
                      ["warm_start", "verbose", "oob_score"]}
            params = pd.Series(params)
            model_params[name] = params

        return pd.concat(model_params)

class Trainer:
    def __init__(self, model: PycaretInterface, training_output: str="training_results", num_splits: int=5, 
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
        self.output_path = Path(training_output)  # for the model results
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.experiment = model
        self.log.info(f"Test_size: {test_size}")
        self.log.info(f"Outliers: {', '.join(self.outliers)}")
        self.log.info(f"Scaler: {scaler}")
        self.log.info(f"Number of kfolds: {self.num_splits}")
        self.log.info(f"The output path: {self.output_path}")
    
    def return_train_test(self, train_index, test_index, features = None, strategy="holdout", split_index=None):
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

    def rank_results(self, results, returned_models, scoring_function):

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
        top_params = self.experiment.get_params(sorted_models)

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
        using cross-validation but only on the training data. It is a nested cross-validation

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

        return pd.concat(res, axis=1), mod, top_params
    
    def setup_holdout(self, X_train, X_test, scoring_fn, plot=()):
        transformed_x, test_x = self._scale(X_train, X_test)
        sorted_results, sorted_models, top_params = self.analyse_models(transformed_x, test_x, scoring_fn)
        if plot:
            self.experiment.plots = plot
            self.experiment.plot_best_models(sorted_models)
        return sorted_results, sorted_models, top_params