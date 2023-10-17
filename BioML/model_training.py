from typing import Iterable
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, ShuffleSplit
import pandas as pd
from BioML import features
from BioML.utilities import scale, write_excel, Log
from pathlib import Path
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
import argparse
import numpy as np
from sklearn.metrics import average_precision_score
import time
from dataclasses import dataclass, field


def arg_parse():
    parser = argparse.ArgumentParser(description="Train the models")

    parser.add_argument("-o", "--training_output", required=False,
                        help="The path where to save the models training results",
                        default="training_results")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format")
    parser.add_argument("-n", "--num_thread", required=False, default=50, type=int,
                        help="The number of threads to search for the hyperparameter space")
    parser.add_argument("-s", "--scaler", required=False, default="robust", choices=("robust", "standard", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to RobustScaler")
    parser.add_argument("-e", "--excel", required=False,
                        help="The file to where the selected features are saved in excel format",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-k", "--kfold_parameters", required=False,
                        help="The parameters for the kfold in num_split:test_size format", default="5:0.2")
    parser.add_argument("-ot", "--outliers", nargs="+", required=False, default=(),
                        help="A list of outliers if any, the name should be the same as in the excel file with the "
                             "filtered features, you can also specify the path to a file in plain text format, each "
                             "record should be in a new line")
    parser.add_argument("-bu", "--budget_time", required=False, default=None, type=float,
                        help="The time budget for the training in minutes, should be > 0 or None")
    parser.add_argument("-pw", "--precision_weight", required=False, default=1.2, type=float,
                        help="Weights to specify how relevant is the precision for the ranking of the different "
                             "features")
    parser.add_argument("-rw", "--recall_weight", required=False, default=0.8, type=float,
                        help="Weights to specify how relevant is the recall for the ranking of the different features")

    parser.add_argument("-rpw", "--report_weight", required=False, default=0.6, type=float,
                        help="Weights to specify how relevant is the f1, precision and recall for the ranking of the "
                             "different features with respect to MCC which is a more general measures of "
                             "the performance of a model")
    parser.add_argument("-dw", "--difference_weight", required=False, default=1.2, type=float,
                        help="How important is to have similar training and test metrics")
    parser.add_argument("-r2", "--r2_weight", required=False, default=0.8, type=float,
                        help="The weights for the R2 score")
    parser.add_argument("-st", "--strategy", required=False, choices=("holdout", "kfold"), default="holdout",
                        help="The spliting strategy to use")
    parser.add_argument("-pr", "--problem", required=False, choices=("classification", "regression"), 
                        default="classification", help="Classification or Regression problem")
    parser.add_argument("-be", "--best_model", required=False, default=3, type=int,
                        help="The number of best models to select, it affects the analysis and the save hyperparameters")
    parser.add_argument("--seed", required=False, default=None, type=int, help="The seed for the random state")


    args = parser.parse_args()

    return [args.label, args.training_output, args.budget_time, args.num_thread, args.scaler,
            args.excel, args.kfold_parameters, args.outliers, args.precision_weight, args.recall_weight,
            args.report_weight, args.difference_weight, args.r2_weight, args.strategy, args.problem, args.best_model,
            args.seed]


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
        

class Trainer:
    def __init__(self, training_output: str="training_results", num_splits: int=5, test_size: float=0.2,
                 outliers: tuple[str, ...]=(), scaler: str="robust"):
        
        """
        Initialize a Trainer object with the given parameters.

        Parameters
        ----------
        training_output : str, optional
            The path to the directory where the training results will be saved. Defaults to "training_results".
        num_splits : int, optional
            The number of splits to use in cross-validation. Defaults to 5.
        test_size : float, optional
            The proportion of the dataset to include in the test split. Defaults to 0.2.
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
    
    def _scale(self, train_index, test_index, features = None, strategy="holdout", split_index=None):
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

    def setup_classifier(self, X_train, X_test):

        self.model.setup(data=X_train, target=self.label_name, normalize=False, preprocess=False, 
                         log_experiment=False, experiment_name="Classification", 
                        session_id = self.seed, fold_shuffle=True, fold=10, test_data=X_test)
        
        self.model.add_metric("averagePre", "Average Precision Score", average_precision_score, 
                                   average="weighted", target="pred_proba", multiclass=False)

        return self.classifier
    
    def setup_regressor(self, X_train, X_test):

        self.model.setup(data=X_train, target=self.label_name, normalize=False, preprocess=False, log_experiment=False, experiment_name="Regression", 
                              session_id = self.seed, fold_shuffle=True, fold=10, test_data=X_test)
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
    
    def analyse_best_models(self, sorted_models, split_ind=None):
        
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

    
    def get_params(self, sorted_models):
        model_params = {}
        for ind, (name, model) in enumerate(sorted_models.items(), 1):
            if ind > self.best_model:
                break
            if name != "catboost":
                params = model.get_params()
            else:
                params = model.get_all_params()
            params = {key: value for key, value in params.items() if key not in ["warm_start", "verbose",
                                                                                            "oob_score"]}
            params = pd.Series(model_params)
            model_params[name] = params

        return model_params
    
class Classifier(Trainer):
    def __init__(self, model: PycaretInterface, training_output="training_results", num_splits=5, test_size=0.2,
                 outliers: tuple[str, ...]=(), scaler="robust",  ranking_params=None,  
                 drop: tuple[str] = ("ada", "gpc", "lightgbm")):
        # initialize the Trainer class
        super().__init__(training_output, num_splits, test_size, outliers, scaler)
        # change the ranking parameters
        ranking_dict = dict(precision_weight=1.2, recall_weight=0.8, report_weight=0.6, 
                            difference_weight=1.2)
        if isinstance(ranking_params, dict):
            for key, value in ranking_params.items():
                if key not in ranking_dict:
                    raise KeyError(f"The key {key} is not found in the ranking params use theses keys: {', '.join(ranking_dict.keys())}")
                ranking_dict[key] = value

        self.classifier = model
        self.classifier.final_models = [x for x in self.classifier.final_models if x not in drop]
        self.pre_weight = ranking_dict["precision_weight"]
        self.rec_weight = ranking_dict["recall_weight"]
        self.report_weight = ranking_dict["report_weight"]
        self.difference_weight = ranking_dict["difference_weight"]

    def train_classifier(self, X_train, X_test):

        # To access the transformed data
        self.classifier.setup_classifier(X_train, X_test)
        
        results, returned_models = self.classifier.train()
        
        return results, returned_models
    
    def _calculate_score_dataframe(self, dataframe):
        cv_train = dataframe.loc[("CV-Train", "Mean")]
        cv_val = dataframe.loc[("CV-Val", "Mean")]

        mcc = ((cv_train["MCC"] + cv_val["MCC"])
                - self.difference_weight * abs(cv_val["MCC"] - cv_val["MCC"] ))
        
        prec = ((cv_train["Prec."] + cv_val["Prec."])
                - self.difference_weight * abs(cv_val["Prec."] - cv_train["Prec."]))
        
        recall = ((cv_train["Recall"] + cv_val["Recall"])
                - self.difference_weight * abs(cv_val["Recall"] - cv_train["Recall"]))
        
        return mcc + self.report_weight * (self.pre_weight * prec + self.rec_weight * recall)
    
    def setup_holdout(self, feature, plot=("learning", "confusion_matrix", "class_report")):
        """
        A function that splits the data into training and test sets and then trains the models
        using cross-validation but only on the training data

        Parameters
        ----------
        feature : pd.DataFrame
            A dataframe containing the training samples and the features
        plot : bool, optional
            Plot the plots relevant to the models, by default 1, 4 and 5
                1. learning: learning curve
                2. pr: Precision recall curve
                3. auc: the ROC curve
                4. confusion_matrix 
                5. class_report: read classification_report from sklearn.metrics

        Returns
        -------
        dict[pd.DataFrame]
            A dictionary with the sorted results from pycaret
        dict[models]
            A dictionary with the sorted models from pycaret
         
        """
        X_train, X_test = train_test_split(feature, test_size=self.test_size, random_state=self.seed, stratify=feature[self.label])
        transformed_x, test_x = self._scale(X_train, X_test)
        results, returned_models = self.train_classifier(transformed_x, test_x)
        sorted_results, sorted_models = self.rank_results(results, returned_models, self._calculate_score_dataframe)
        top_params = self._get_params(sorted_models)
        if plot:
            self.classifier.plots = plot
            self.classifier.analyse_best_models(sorted_models)
        return sorted_results, sorted_models, top_params

    def setup_kfold(self, feature, plot=()):
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
        dict[tuple(dict[pd.DataFrame], dict[models]))]
            A dictionary with the sorted results and sorted models from pycaret organized by split index or kfold index
        """
        res = {}
        top_params = {}
        mod = {}
        skf = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.seed)
        for ind, (train_index, test_index) in enumerate(skf.split(feature, feature[self.label])):
            transformed_x, test_x = self._scale(feature, train_index, test_index, strategy="kfold", split_index=ind)
            results, returned_models = self.train_classifier(transformed_x, test_x)
            sorted_results, sorted_models = self.rank_results(results, returned_models, self._calculate_score_dataframe)
            if plot:
                self.analyse_best_models(self.classifier, sorted_models, [x for x in self.classification_plots if x in plot], split_ind=ind)
            res[f"split_{ind}"] = sorted_results
            top_params[f"split_{ind}"] = self._get_params(sorted_models)
            mod[f"split_{ind}"] = sorted_models
        return pd.concat(res, axis=1), mod, pd.concat(top_params, axis=1)
    
    def run(self, features, strategy="holdout", plot_holdout=("learning", "confusion_matrix", "class_report"), 
            plot_kfold=()):
        for key, value in features.items():
            if strategy == "holdout":
                sorted_results, sorted_models, top_params = self.setup_holdout(value, plot_holdout)
                write_excel(self.output_path / "training_results.xlsx", sorted_results, key)
                write_excel(self.output_path / f"top_{self.best_model}_hyperparameters.xlsx", top_params, key)
            elif strategy == "kfold":
                sorted_results, sorted_models, top_params = self.setup_kfold(value, plot_kfold)
                write_excel(self.output_path / "training_results.xlsx", sorted_results, f"{key}")
                write_excel(self.output_path / f"top_{self.best_model}_hyperparameters.xlsx", top_params, f"{key}")
            else:
                raise ValueError("strategy should be either holdout or kfold")
            
        return sorted_results, sorted_models, top_params


class Regressor(Trainer):
    def __init__(self, model: PycaretInterface, training_output="training_results", num_splits=5, test_size=0.2,
                 outliers=(), scaler="robust", ranking_params=None, drop=("tr", "kr", "ransac", "ard", "ada", "lightgbm")):

        super().__init__(training_output, num_splits, test_size, outliers, scaler)
        
        ranking_dict = dict(R2_weight=0.8, difference_weight=1.2)
        if isinstance(ranking_params, dict):
            for key, value in ranking_params.items():
                if key not in ranking_dict:
                    raise KeyError(f"The key {key} is not  found in the ranking params use theses keys: {', '.join(ranking_dict.keys())}")
                ranking_dict[key] = value

        self.regressor = model
        self.regressor.final_models = [x for x in self.regressor.final_models if x not in drop]
        self.regression_plots = ["residuals", "error", "learning"]
        self.difference_weight = ranking_dict["difference_weight"]
        self.R2_weight = ranking_dict["R2_weight"]

    def _calculate_score_dataframe(self, dataframe):
        cv_train = dataframe.loc[("CV-Train", "Mean")]
        cv_val = dataframe.loc[("CV-Val", "Mean")]

        rmse = ((cv_train["RMSE"] + cv_val["RMSE"])
                - self.difference_weight * abs(cv_val["RMSE"] - cv_val["RMSE"] ))
        
        r2 = ((cv_train["R2"] + cv_val["R2"])
                - self.difference_weight * abs(cv_val["R2"] - cv_train["R2"]))
        
        
        return rmse + (self.R2_weight * r2)

    def train_regressor(self, X_train, X_test):

        self.regressor.setup_regressor(X_train, X_test)
        # To access the transformed data
        results, returned_models = self.regressor.train()
    
        return results, returned_models
    
    def setup_holdout(self, feature, plot=("residuals", "error", "learning")):
        """
        A function that splits the data into training and test sets and then trains the models
        using cross-validation but only on the training data

        Parameters
        ----------
        feature : pd.DataFrame
            A dataframe containing the training samples and the features
        plot : bool, optional
            Plot the plots relevant to the models, by default 1,2,3
                1. residuals: Plots the difference (predicted-actual value) vs predicted value for train and test
                2. error: Plots the actual values vs predicted values
                3. learning: learning curve

        Returns
        -------
        dict[pd.DataFrame]
            A dictionary with the sorted results from pycaret
        dict[models]
            A dictionary with the sorted models from pycaret
         
        """
        X_train, X_test = train_test_split(feature, test_size=self.test_size, random_state=self.seed, stratify=feature[self.label])
        transformed_x, test_x = self._scale(X_train, X_test)
        results, returned_models = self.train_regressor(transformed_x, test_x)
        sorted_results, sorted_models = self.rank_results(results, returned_models, self._calculate_score_dataframe)
        top_params = self._get_params(sorted_models)
        if plot:
            self.analyse_best_models(self.regression, sorted_models, [x for x in self.regression_plots if x in plot])
        return sorted_results, sorted_models, top_params

    def setup_kfold(self, feature, plot=()):
        """
        A function that splits the data into kfolds of training and test sets and then trains the models
        using cross-validation but only on the training data. It is a nested cross-validation

        Parameters
        ----------
        feature : pd.DataFrame
            A dataframe containing the training samples and the features
        plot : bool, optional
            Plot the plots relevant to the models, by default -> ()
                1. residuals: Plots the difference (predicted-actual value) vs predicted value for train and test
                2. error: Plots the actual values vs predicted values
                3. learning: learning curve

        Returns
        -------
        dict[tuple(dict[pd.DataFrame], dict[models]))]
            A dictionary with the sorted results and sorted models from pycaret organized by split index or kfold index
        """
        res = {}
        top_params = {}
        mod = {}
        skf = ShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.seed)
        for ind, (train_index, test_index) in enumerate(skf.split(feature, feature[self.label])):
            transformed_x, test_x = self._scale(feature, train_index, test_index, strategy="kfold", split_index=ind)
            results, returned_models = self.train_regressor(transformed_x, test_x)
            sorted_results, sorted_models = self.rank_results(results, returned_models, self._calculate_score_dataframe)
            if plot:
                self.analyse_best_models(self.regression, sorted_models, [x for x in self.regression_plots if x in plot], split_ind=ind)
            res[f"split_{ind}"] = sorted_results
            top_params[f"split_{ind}"] = self._get_params(sorted_models)
            mod[f"split_{ind}"] = sorted_models
        return pd.concat(res, axis=1), mod, pd.concat(top_params, axis=1)
    
    def run(self, features, strategy="holdout", plot_holdout=("residuals", "error", "learning"), plot_kfold=()):
        for key, value in features.items():
            if strategy == "holdout":
                sorted_results, sorted_models, top_params = self.setup_holdout(value, plot_holdout)
                write_excel(self.output_path / "training_results.xlsx", sorted_results, key)
                write_excel(self.output_path / f"top_{self.best_model}_hyperparameters.xlsx", top_params, key)
            elif strategy == "kfold":
                sorted_results, sorted_models, top_params = self.setup_kfold(value, plot_kfold)
                write_excel(self.output_path / "training_results.xlsx", sorted_results, f"{key}")
                write_excel(self.output_path / f"top_{self.best_model}_hyperparameters.xlsx", top_params, f"{key}")
            else:
                raise ValueError("strategy should be either holdout or kfold")
        
        return sorted_results, sorted_models, top_params


def main():
    label, training_output, trial_time, num_thread, scaler, excel, kfold, outliers, \
        precision_weight, recall_weight, report_weight, difference_weight, r2_weight, strategy, problem, seed, \
    best_model = arg_parse()
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])

    if len(outliers) > 0 and Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = [x.strip() for x in out.readlines()]
    if problem == "classification":
        ranking_dict = dict(precision_weight=precision_weight, recall_weight=recall_weight, report_weight=report_weight, 
                            difference_weight=difference_weight)
        training = Classifier(excel, label, training_output, num_split, test_size, outliers, scaler, trial_time, 
                              num_thread, ranking_dict, best_model, seed)
    elif problem == "regression":
        ranking_dict = dict(R2_weight=r2_weight, difference_weight=difference_weight)
        training = Regressor(excel, label, training_output, num_split, test_size, outliers, scaler, trial_time, 
                             num_thread, ranking_dict, best_model, seed)
    
    training.run(strategy)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
