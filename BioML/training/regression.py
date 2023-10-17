from re import X
from typing import Iterable
from sklearn.model_selection import train_test_split, ShuffleSplit
import pandas as pd
from BioML import features
from BioML.utilities import write_excel
from pathlib import Path
import argparse
import numpy as np
from .base import PycaretInterface, Trainer, DataParser
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
                        help="The file to where the selected or training features are saved in excel format",
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

    parser.add_argument("-d", "--drop", nargs="+", required=False, default=(), help="The models to drop")

    args = parser.parse_args()

    return [args.label, args.training_output, args.budget_time, args.num_thread, args.scaler,
            args.excel, args.kfold_parameters, args.outliers, args.precision_weight, args.recall_weight,
            args.report_weight, args.difference_weight, args.r2_weight, args.strategy, args.problem, args.best_model,
            args.seed, args.drop]



class Regressor(Trainer):
    def __init__(self, model: PycaretInterface, training_output="training_results", num_splits=5, test_size=0.2,
                 outliers=(), scaler="robust", ranking_params=None, drop=("tr", "kr", "ransac", "ard", "ada", "lightgbm")):

        super().__init__(model, training_output, num_splits, test_size, outliers, scaler)
        
        ranking_dict = dict(R2_weight=0.8, difference_weight=1.2)
        if isinstance(ranking_params, dict):
            for key, value in ranking_params.items():
                if key not in ranking_dict:
                    raise KeyError(f"The key {key} is not  found in the ranking params use theses keys: {', '.join(ranking_dict.keys())}")
                ranking_dict[key] = value

        self.experiment.final_models = [x for x in self.experiment.final_models if x not in drop]
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
    
    def run_holdout(self, feature, plot=("residuals", "error", "learning")):
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
        self.log.info("------ Running holdout -----")
        X_train, X_test = train_test_split(feature.features, test_size=self.test_size, random_state=self.experiment.seed)
        sorted_results, sorted_models, top_params = self.setup_holdout(X_train, X_test, self._calculate_score_dataframe, plot)
        return sorted_results, sorted_models, top_params

    def run_kfold(self, feature: DataParser, plot=()):
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
        self.log.info("------ Running kfold -----")
        skf = ShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.experiment.seed)
        sorted_results, sorted_models, top_params = self.setup_kfold(feature.features, feature.label, skf, plot, feature.with_split)
        return sorted_results, sorted_models, top_params

def run(output_path, best_model, features,  sorted_results, sorted_models, top_params, strategy="holdout", 
        plot_holdout=("learning", "confusion_matrix", "class_report"), 
        plot_kfold=()):
    for key, value in features.items():
        if strategy == "holdout":
            sorted_results, sorted_models, top_params = self.setup_holdout(value, plot_holdout)
            write_excel(output_path / "training_results.xlsx", sorted_results, key)
            write_excel(output_path / f"top_{best_model}_hyperparameters.xlsx", top_params, key)
        elif strategy == "kfold":
            sorted_results, sorted_models, top_params = self.setup_kfold(value, plot_kfold)
            write_excel(output_path / "training_results.xlsx", sorted_results, f"{key}")
            write_excel(output_path / f"top_{best_model}_hyperparameters.xlsx", top_params, f"{key}")
        else:
            raise ValueError("strategy should be either holdout or kfold")
        
    return sorted_results, sorted_models, top_params

def main():
    label, training_output, trial_time, num_thread, scaler, excel, kfold, outliers, \
        precision_weight, recall_weight, report_weight, difference_weight, r2_weight, strategy, problem, seed, \
    best_model, drop = arg_parse()
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])

    if len(outliers) > 0 and Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = [x.strip() for x in out.readlines()]
    feature = DataParser(label, excel)
    experiment = PycaretInterface(problem, feature.label, seed, budget_time=trial_time, best_model=best_model)

    if problem == "classification":
        ranking_dict = dict(precision_weight=precision_weight, recall_weight=recall_weight, report_weight=report_weight, 
                            difference_weight=difference_weight)
        training = Classifier(experiment, training_output, num_split, test_size,
                                outliers, scaler, ranking_dict, drop)
    elif problem == "regression":
        ranking_dict = dict(R2_weight=r2_weight, difference_weight=difference_weight)
        training = Regressor(experiment, training_output, num_split, test_size, outliers, scaler, ranking_dict, drop)
    
    if strategy == "holdout":
        sorted_results, sorted_models, top_params = training.run_holdout(feature)
    elif strategy == "kfold":
        sorted_results, sorted_models, top_params = training.run_kfold(feature)

    write_excel(output_path / "training_results.xlsx", sorted_results, f"{key}")
    write_excel(output_path / f"top_{best_model}_hyperparameters.xlsx", top_params, f"{key}")

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
