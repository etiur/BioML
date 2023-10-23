from sklearn.model_selection import train_test_split, ShuffleSplit
from pathlib import Path
import argparse
from .base import PycaretInterface, Trainer, DataParser, write_results
import pandas as pd
from collections import defaultdict


def arg_parse():
    parser = argparse.ArgumentParser(description="Train Regression models")

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
    parser.add_argument("-dw", "--difference_weight", required=False, default=1.2, type=float,
                        help="How important is to have similar training and test metrics")
    parser.add_argument("-r2", "--r2_weight", required=False, default=0.8, type=float,
                        help="The weights for the R2 score")
    parser.add_argument("-st", "--strategy", required=False, choices=("holdout", "kfold"), default="holdout",
                        help="The spliting strategy to use")
    parser.add_argument("-be", "--best_model", required=False, default=3, type=int,
                        help="The number of best models to select, it affects the analysis and the save hyperparameters")
    parser.add_argument("--seed", required=False, default=None, type=int, help="The seed for the random state")

    parser.add_argument("-d", "--drop", nargs="+", required=False, default=(), help="The models to drop")
    parser.add_argument("--tune", action="store_true", required=False, default=False, 
                        help="If to tune the best models")
    
    parser.add_argument("-se", "--select", required=False, default=None, 
                        help="what model to chose", choices=("stacked", "majority", "best"))
    parser.add_argument("-i", "--index", required=False, default=None, type=int, 
                        help="which one of the best models to choose, starting from 0, only use when select = best")
    
    args = parser.parse_args()

    return [args.label, args.training_output, args.budget_time, args.num_thread, args.scaler,
            args.excel, args.kfold_parameters, args.outliers,
            args.difference_weight, args.r2_weight, args.strategy, args.best_model,
            args.seed, args.drop, args.tune, args.select, args.index]


class Regressor(Trainer):
    def __init__(self, model: PycaretInterface, output="training_results", num_splits=5, test_size=0.2,
                 outliers=(), scaler="robust", ranking_params=None, drop=("tr", "kr", "ransac", "ard", "ada", "lightgbm")):

        super().__init__(model, output, num_splits, test_size, outliers, scaler)
        
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
    
    def retune_best_models(self, sorted_models: dict, optimize: str = "RMSE", num_iter: int = 5):
        if "split" in list(sorted_models)[0]:
            new_models = {}
            new_results = {}
            new_params = {}
            for key, sorted_model_by_split in sorted_models.items():
                new_results[key], new_models[key], new_params[key] = self._retune_best_models(sorted_model_by_split, optimize, num_iter)
            return pd.concat(new_results, axis=1), new_models, pd.concat(new_params)
        return self._retune_best_models(sorted_models, optimize, num_iter)
    
    def stack_models(self, sorted_models: dict, optimize="RMSE", probability_theshold: float = 0.5, meta_model=None):

        return self._stack_models(sorted_models, optimize, probability_theshold, meta_model=meta_model)
    
    def create_majority_model(self, sorted_models: dict, optimize: str = "RMSE", probability_theshold: float = 0.5, weights=None):
    
        return self._create_majority_model(sorted_models, optimize, probability_theshold, weights)
    
    def finalize_model(self, sorted_model):
        return self._finalize_model(sorted_model)
    
    def save_model(self, sorted_models, filename: str | dict[str, str] | None = None):
        return self._save_model(sorted_models, filename)
    




def main():
    label, training_output, trial_time, num_thread, scaler, excel, kfold, outliers, \
        difference_weight, r2_weight, strategy, seed, best_model, drop, tune = arg_parse()
    
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])
    training_output = Path(training_output)
    if len(outliers) > 0 and Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = [x.strip() for x in out.readlines()]

    
    feature = DataParser(label, excel)
    experiment = PycaretInterface("regression", feature.label, seed, budget_time=trial_time, best_model=best_model)

    ranking_dict = dict(R2_weight=r2_weight, difference_weight=difference_weight)
    training = Regressor(experiment, training_output, num_split, test_size, outliers, scaler, 
                         ranking_dict, drop)
    
    if strategy == "holdout":
        sorted_results, sorted_models, top_params = training.run_holdout(feature)
    elif strategy == "kfold":
        sorted_results, sorted_models, top_params = training.run_kfold(feature)

    # saving the results in a dictionary and writing it into excel files
    results = defaultdict(dict)
    results["not_tuned"][strategy] = sorted_results, sorted_models, top_params
    results["not_tuned"]["stacked"] = training.stack_models(sorted_models)
    results["not_tuned"]["majority"] = training.create_majority_model(sorted_models)

    if tune:
        sorted_result_tune, sorted_models_tune, top_params_tune = training.retune_best_models(sorted_models)
        results["tuned"][strategy] = sorted_result_tune, sorted_models_tune, top_params_tune
        results["tuned"]["stacked"] = training.stack_models(sorted_models_tune)
        results["tuned"]["majority"] = training.create_majority_model(sorted_models_tune)

    for tune_status, result_dict in results.items():
        for key, value in result_dict.items():
            if len(value) == 2:
                write_results(training_output/f"{tune_status}", value[0], sheet_name=key)
            elif len(value) == 3:
                write_results(training_output/f"{tune_status}", value[0], value[2], sheet_name=key)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
