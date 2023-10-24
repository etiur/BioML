from sklearn.model_selection import train_test_split, ShuffleSplit
from pathlib import Path
import argparse

from BioML import features
from .base import PycaretInterface, Trainer, DataParser, write_results, generate_training_results
import pandas as pd


def arg_parse():
    parser = argparse.ArgumentParser(description="Train Regression models")

    parser.add_argument("-o", "--training_output", required=False,
                        help="The path where to save the models training results",
                        default="training_results")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format of string if it is insde the features")
    parser.add_argument("-n", "--num_thread", required=False, default=50, type=int,
                        help="The number of threads to search for the hyperparameter space")
    parser.add_argument("-s", "--scaler", required=False, default="robust", choices=("robust", "standard", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to RobustScaler")
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
    parser.add_argument("-r2", "--r2_weight", required=False, default=0.8, type=float,
                        help="The weights for the R2 score")
    parser.add_argument("-st", "--strategy", required=False, choices=("holdout", "kfold"), default="holdout",
                        help="The spliting strategy to use")
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

    args = parser.parse_args()

    return [args.label, args.training_output, args.budget_time, args.scaler,
            args.excel, args.kfold_parameters, args.outliers,
            args.difference_weight, args.r2_weight, args.strategy, args.best_model,
            args.seed, args.drop, args.tune, args.plot, args.optimize, args.selected]


class Regressor:
    def __init__(self, ranking_params=None, drop=("tr", "kr", "ransac", "ard", "ada", "lightgbm"), selected=None):
        
        ranking_dict = dict(R2_weight=0.8, difference_weight=1.2)
        if isinstance(ranking_params, dict):
            for key, value in ranking_params.items():
                if key not in ranking_dict:
                    raise KeyError(f"The key {key} is not  found in the ranking params use theses keys: {', '.join(ranking_dict.keys())}")
                ranking_dict[key] = value

        self.drop = drop
        self.difference_weight = ranking_dict["difference_weight"]
        self.R2_weight = ranking_dict["R2_weight"]
        self.selected = selected

    def _calculate_score_dataframe(self, dataframe):
        cv_train = dataframe.loc[("CV-Train", "Mean")]
        cv_val = dataframe.loc[("CV-Val", "Mean")]

        rmse = ((cv_train[self.optimize] + cv_val[self.optimize])
                - self.difference_weight * abs(cv_val[self.optimize] - cv_val[self.optimize] ))
        
        r2 = ((cv_train["R2"] + cv_val["R2"])
                - self.difference_weight * abs(cv_val["R2"] - cv_train["R2"]))
        
        
        return rmse + (self.R2_weight * r2)
    
    def run_training(self, trainer: Trainer, feature: DataParser, plot=("residuals", "error", "learning")):
        """
        A function that splits the data into training and test sets and then trains the models
        using cross-validation but only on the training data

        Parameters
        ----------
        feature : pd.DataFrame
            A dataframe containing the training samples and the features
        plot : bool, optional
            Plot the plots relevant to the models, by default all of them
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
        X_train, X_test, y_train, y_test = train_test_split(feature.features, feature.label, test_size=self.test_size, random_state=self.experiment.seed)
        transformed_x, test_x = feature.scale(X_train, X_test)
        transformed_x, test_x = feature.process(X_train, X_test, y_train, y_test)
        sorted_results, sorted_models, top_params = trainer.setup_training(transformed_x, test_x, self._calculate_score_dataframe,
                                                                           plot, drop=self.drop, selected=self.selected)
        return sorted_results, sorted_models, top_params
    
    
def main():
    label, training_output, trial_time, scaler, excel, kfold, outliers, \
        difference_weight, r2_weight, strategy, seed, best_model, drop, tune, plot, optimize, selected = arg_parse()
    
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])
    training_output = Path(training_output)
    if outliers and Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = [x.strip() for x in out.readlines()]

    outliers = {"x_train": outliers, "x_test": outliers}
    
    feature = DataParser(excel, label,  outliers=outliers, scaler=scaler)
    experiment = PycaretInterface("regression", feature.label, seed, budget_time=trial_time, best_model=best_model, 
                                  output_path=training_output)

    ranking_dict = dict(R2_weight=r2_weight, difference_weight=difference_weight)
    training = Trainer(experiment, num_split, test_size)
    regressor = Regressor(ranking_dict, drop, selected=selected)
    
    results = generate_training_results(regressor, training, feature, plot, optimize, tune, strategy)

    for tune_status, result_dict in results.items():
        predictions = []
        for key, value in result_dict.items():
            # get the test set prediction results
            predictions.append(training.predict_on_test_set(value[1], key))
            # write the results on excel files
            if len(value) == 2:
                write_results(training_output/f"{tune_status}", value[0], sheet_name=key)
            elif len(value) == 3:
                write_results(training_output/f"{tune_status}", value[0], value[2], sheet_name=key)
            
        write_results(training_output/f"{tune_status}", pd.concat(predictions), sheet_name=f"test_results")


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
