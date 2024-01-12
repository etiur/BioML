"""
A module that performs regression analysis on a dataset.
"""
from pathlib import Path
import argparse
from functools import partial
from typing import Iterable
import pandas as pd
from .base import PycaretInterface, Trainer, DataParser
from ..utilities.training import evaluate_all_models, write_results
from ..utilities import split_methods as split


def arg_parse():
    parser = argparse.ArgumentParser(description="Train Regression models")

    parser.add_argument("-o", "--training_output", required=False,
                        help="The path where to save the models training results",
                        default="training_results")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format of string if it is insde the features")
    parser.add_argument("-n", "--num_thread", required=False, default=50, type=int,
                        help="The number of threads to search for the hyperparameter space")
    parser.add_argument("-s", "--scaler", required=False, default="zscore", choices=("robust", "zscore", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to zscore")
    parser.add_argument("-i", "--training_features", required=True,
                        help="The file to where the training features are saved in excel or csv format")
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
    parser.add_argument("-be", "--best_model", required=False, default=3, type=int,
                        help="The number of best models to select, it affects the analysis and the save hyperparameters")
    parser.add_argument("--seed", required=False, default=None, type=int, help="The seed for the random state")

    parser.add_argument("-d", "--drop", nargs="+", required=False, default=("tr", "kr", "ransac", "ard", "ada", "lightgbm"), 
                        choices=('lr','lasso','ridge','en','lar','llar','omp','br','ard','par','ransac',
                                  'tr','huber','kr','svm','knn','dt','rf','et','ada','gbr','mlp','xgboost',
                                  'lightgbm','catboost','dummy'), help="The models to drop")
    parser.add_argument("-se", "--selected", nargs="+", required=False, default=None, 
                        choices=('lr','lasso','ridge','en','lar','llar','omp','br','ard','par','ransac',
                                  'tr','huber','kr','svm','knn','dt','rf','et','ada','gbr','mlp','xgboost',
                                  'lightgbm','catboost','dummy'), help="The models to select, when None almost all models are selected")
    parser.add_argument("--tune", action="store_true", required=False, help="If to tune the best models")
    parser.add_argument("-op", "--optimize", required=False, default="RMSE", 
                        choices=("RMSE", "R2", "MSE", "MAE", "RMSLE", "MAPE"), help="The metric to optimize")
    parser.add_argument("-p", "--plot", nargs="+", required=False, default=("residuals", "error", "learning"),
                        help="The plots to show")
    parser.add_argument("-sh", "--sheet_name", required=False, default=None, 
                        help="The sheet name for the excel file if the training features is in excel format")
    parser.add_argument("-ni", "--num_iter", default=30, type=int, required=False, 
                        help="The number of iterations for the hyperparameter search")
    parser.add_argument("-st", "--split_strategy", required=False, default="random", 
                        choices=("mutations", "cluster", "random"), help="The strategy to split the data")
    parser.add_argument("-c", "--cluster", required=False, default=None, help="The path to the cluster file generated by mmseqs2")
    parser.add_argument("-m", "--mutations", required=False, default=None, help="The column name of the mutations in the training data")
    parser.add_argument("-tnm", "--test_num_mutations", required=False, default=None, type=int, 
                        help="The threshold of number of mutations to be included in the test set")
    parser.add_argument("-g", "--greater", required=False, action="store_false", 
                        help="Include in the test set, mutations that are greater of less than the threshold, default greater")

    args = parser.parse_args()

    return [args.label, args.training_output, args.budget_time, args.scaler,
            args.training_features, args.kfold_parameters, args.outliers, args.difference_weight, 
            args.best_model, args.seed, args.drop, args.tune, args.plot, args.optimize, 
            args.selected, args.sheet_name, args.num_iter, args.split_strategy, args.cluster, 
            args.mutations, args.test_num_mutations, args.greater]


class Regressor:
    """
    A class that performs regression analysis on a dataset.

    Attributes
    ----------
    ranking_params : dict, optional
        A dictionary containing the ranking parameters to use for feature selection. Default is None.
    drop : tuple, optional
        A tuple of strings containing the names of the models to drop during feature selection. Default is 
        ("tr", "kr", "ransac", "ard", "ada", "lightgbm").
    selected : list, optional
        A list of strings containing the names of the selected features. Default is None.
    test_size : float, optional
        The proportion of the dataset to use for testing. Default is 0.2.
    optimize : str, optional
        The name of the optimization metric to use for regression analysis. Default is "RMSE".
    plot : bool, optional
        Plot the plots relevant to the models, by default all of them
            1. residuals: Plots the difference (predicted-actual value) vs predicted value for train and test
            2. error: Plots the actual values vs predicted values
            3. learning: learning curve

    """
    def __init__(self, ranking_params: dict[str, float] | None=None, 
                 drop: Iterable[str]=("tr", "kr", "ransac", "ard", "ada", "lightgbm"), selected: Iterable[str]=(), 
                 test_size: float = 0.2, optimize: str="RMSE", plot: Iterable[str]=("residuals", "error", "learning")):
        
        ranking_dict = dict(difference_weight=1.2)
        if isinstance(ranking_params, dict):
            for key, value in ranking_params.items():
                if key not in ranking_dict:
                    raise KeyError(f"The key {key} is not  found in the ranking params use theses keys: {', '.join(ranking_dict.keys())}")
                ranking_dict[key] = value
        self.test_size = test_size
        self.drop = drop
        self.difference_weight = ranking_dict["difference_weight"]
        self.selected = selected
        self.optimize = optimize
        self.plot = plot

    def _calculate_score_dataframe(self, dataframe: pd.DataFrame) -> float | int: # type: ignore
        """
        Calculates the score of a DataFrame based on the optimization metric and R2 score.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame containing the performance metrics of a regression model.

        Returns
        -------
        int or float
            The calculated score of the DataFrame.


        Examples
        --------
        >>> dataframe = pd.read_csv("performance_metrics.csv")
        >>> score = _calculate_score_dataframe(dataframe)
        ... # Calculates the score of the DataFrame based on the optimization metric and R2 score.
        """
        cv_train = dataframe.loc[("CV-Train", "Mean")]
        cv_val = dataframe.loc[("CV-Val", "Mean")]

        if self.optimize != "R2":
            rmse = ((cv_train[self.optimize] + cv_val[self.optimize]) # type: ignore
                    + self.difference_weight * abs(cv_val[self.optimize] - cv_train[self.optimize])) # type: ignore
            
            return - rmse # negative because the sorting is from greater to smaller, in this case the smaller the error value the better

        if self.optimize == "R2":
            r2 = ((cv_train["R2"] + cv_val["R2"]) # type: ignore
                - self.difference_weight * abs(cv_val["R2"] - cv_train["R2"])) # type: ignore
        
            return r2
        
    def sort_holdout_prediction(self, dataframe: pd.DataFrame) -> pd.DataFrame: # type: ignore
        """
        Sorts the predictions of a regression model based on a specified optimization metric and R2 score.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame containing the predictions of the regression model.

        Returns
        -------
        pd.DataFrame
            The sorted DataFrame of predictions.
        """
        if self.optimize == "R2":
            return dataframe.sort_values(self.optimize, ascending=False)
        if self.optimize != "R2":
            return dataframe.sort_values(self.optimize,ascending=True)
    
    
def main():
    label, training_output, trial_time, scaler, excel, kfold, outliers, difference_weight, \
    best_model, seed, drop, tune, plot, optimize, selected, sheet, num_iter, split_strategy, cluster, mutations, \
        test_num_mutations, greater = arg_parse()
    
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])
    training_output = Path(training_output)
    if outliers and Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = tuple(x.strip() for x in out.readlines())
    
    ranking_dict = dict(difference_weight=difference_weight)
    # instantiate all the classes
    # this is only used to read the data
    feature = DataParser(excel, label,  outliers=outliers, sheets=sheet)
    # These are the classes used for regression
    experiment = PycaretInterface("regression", seed, scaler=scaler, budget_time=trial_time, # type: ignore
                                  best_model=best_model, output_path=training_output, optimize=optimize) 
    # this class has the arguments for the trainer for regression 
    regressor = Regressor(ranking_dict, drop, selected=selected, test_size=test_size, optimize=optimize, plot=plot) 

    # It uses the PycaretInterface' models to perform the training but you could use other models as long as it implements the same methods
    training = Trainer(experiment, regressor, num_split, num_iter) # this can be used for classification or regression -> so it is generic
    
    spliting = {"cluster": split.ClusterSpliter(cluster, num_split, random_state=experiment.seed, test_size=test_size),
                "mutations": split.MutationSpliter(mutations, test_num_mutations, greater, 
                                                   num_splits=num_split, random_state=experiment.seed)}
    
    # split the data based on the strategy
    if split_strategy != "random":
        X_train, X_test = spliting[split_strategy].train_test_split(feature.features)

        results, models_dict = training.generate_training_results(X_train, feature.label, tune, 
                                                                  test_data=X_test, fold_strategy=spliting[split_strategy])
    else:
        # train the models and retunr the prediction results
        results, models_dict = training.generate_training_results(feature.features, feature.label, tune)
    
    # sort the results based on the optimization metric
    test_set_predictions = training.generate_holdout_prediction(models_dict)
    evaluate_all_models(experiment.evaluate_model, models_dict, training_output)

    # finally write the results
    for tune_status, result_dict in results.items():
        for key, value in result_dict.items():
            write_results(f"{training_output}/{tune_status}", *value, sheet_name=key)
        write_results(f"{training_output}/{tune_status}", test_set_predictions[tune_status] , sheet_name=f"test_results")


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
