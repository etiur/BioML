import pandas as pd
from .base import PycaretInterface, Trainer, DataParser, write_results
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import argparse
from pathlib import Path
from collections import defaultdict

def arg_parse():
    parser = argparse.ArgumentParser(description="Train classification models")

    parser.add_argument("-o", "--training_output", required=False,
                        help="The path where to save the models training results",
                        default="training_results")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format or string if it is inside training features")
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

    parser.add_argument("-st", "--strategy", required=False, choices=("holdout", "kfold"), default="holdout",
                        help="The spliting strategy to use")

    parser.add_argument("-be", "--best_model", required=False, default=3, type=int,
                        help="The number of best models to select, it affects the analysis and the saved hyperparameters")
    parser.add_argument("--seed", required=False, default=None, type=int, help="The seed for the random state")

    parser.add_argument("-d", "--drop", nargs="+", required=False, default=("ada", "gpc", "lightgbm"),
                        choices=('lr','knn','nb','dt','svm','rbfsvm','gpc','mlp','ridge','rf','qda','ada','gbc',
                                'lda','et','xgboost','lightgbm','catboost','dummy'), 
                        help="The models to drop")
    parser.add_argument("-se", "--selected", nargs="+", required=False, default=None,
                        choices=('lr','knn','nb','dt','svm','rbfsvm','gpc','mlp','ridge','rf','qda','ada','gbc',
                                'lda','et','xgboost','lightgbm','catboost','dummy'), 
                        help="The models to train")

    parser.add_argument("--tune", action="store_false", required=False, default=False, 
                        help="If to tune the best models")
    parser.add_argument("-p", "--plot", nargs="+", required=False, default=("learning", "confusion_matrix", "class_report"),
                        help="The plots to save", choices=("learning", "confusion_matrix", "class_report", "pr", "auc"))
    parser.add_argument("-op", "--optimize", required=False, default="MCC", choices=("MCC", "Prec.", "Recall", "F1", "AUC", "Accuracy", 
                                                                                     "Average Precision Score"),
                        help="The metric to optimize for retuning the best models")

    args = parser.parse_args()

    return [args.label, args.training_output, args.budget_time, args.scaler, args.training_features, args.kfold_parameters, 
            args.outliers, args.precision_weight, args.recall_weight, args.report_weight, args.difference_weight, 
            args.strategy, args.best_model, args.seed, args.drop, args.tune, args.plot, args.optimize, args.selected]


class Classifier(Trainer):
    def __init__(self, model: PycaretInterface, num_splits=5, test_size=0.2,
                 outliers: tuple[str, ...]=(), scaler="robust",  ranking_params: dict[str, float]=None,  
                 drop: tuple[str] = ("ada", "gpc", "lightgbm"), selected=None, optimize="MCC"):
        # initialize the Trainer class
        super().__init__(model, num_splits, test_size, outliers, scaler)
        # change the ranking parameters
        ranking_dict = dict(precision_weight=1.2, recall_weight=0.8, report_weight=0.6, 
                            difference_weight=1.2)
        if isinstance(ranking_params, dict):
            for key, value in ranking_params.items():
                if key not in ranking_dict:
                    raise KeyError(f"The key {key} is not found in the ranking params use theses keys: {', '.join(ranking_dict.keys())}")
                ranking_dict[key] = value
        self.drop = drop
        self.pre_weight = ranking_dict["precision_weight"]
        self.rec_weight = ranking_dict["recall_weight"]
        self.report_weight = ranking_dict["report_weight"]
        self.difference_weight = ranking_dict["difference_weight"]
        self.optimize = optimize
        self.selected = selected
    
    def _calculate_score_dataframe(self, dataframe):
        cv_train = dataframe.loc[("CV-Train", "Mean")]
        cv_val = dataframe.loc[("CV-Val", "Mean")]

        mcc = ((cv_train[self.optimize] + cv_val[self.optimize])
                - self.difference_weight * abs(cv_val[self.optimize] - cv_val[self.optimize] ))
        
        prec = ((cv_train["Prec."] + cv_val["Prec."])
                - self.difference_weight * abs(cv_val["Prec."] - cv_train["Prec."]))
        
        recall = ((cv_train["Recall"] + cv_val["Recall"])
                - self.difference_weight * abs(cv_val["Recall"] - cv_train["Recall"]))
        
        return mcc + self.report_weight * (self.pre_weight * prec + self.rec_weight * recall)
    
    def run_training(self, feature: DataParser, plot: tuple[str, ...]=("learning", "confusion_matrix", "class_report")):
        """
        A function that splits the data into training and test sets and then trains the models
        using cross-validation but only on the training data

        Parameters
        ----------
        feature : DataParser
            A class containing the training samples and the features
        plot : tuple[str, ...], optional
            Plot the plots relevant to the models, by default 1, 4 and 5
                1. learning: learning curve
                2. pr: Precision recall curve
                3. auc: the ROC curve
                4. confusion_matrix 
                5. class_report: read classification_report from sklearn.metrics

        Returns
        -------
        pd.DataFrame
            A dictionary with the sorted results from pycaret
        list[models]
            A dictionary with the sorted models from pycaret
        pd.DataFrame
         
        """
        self.log.info("------ Running holdout -----")
        X_train, X_test = train_test_split(feature.features, test_size=self.test_size, random_state=self.experiment.seed, 
                                           stratify=feature.features[feature.label])
        sorted_results, sorted_models, top_params = self.setup_training(X_train, X_test, self._calculate_score_dataframe, plot, drop=self.drop,
                                                                        selected=self.selected)
        return sorted_results, sorted_models, top_params
    
    def retune_best_models(self, sorted_models: dict, num_iter: int = 5):
        return self._retune_best_models(sorted_models, self.optimize, num_iter)
    
    def stack_models(self, sorted_models: dict, meta_model=None):

        return self._stack_models(sorted_models, self.optimize, meta_model=meta_model)
    
    def create_majority_model(self, sorted_models: dict, weights=None):
    
        return self._create_majority_model(sorted_models, self.optimize, weights)
    
    def finalize_model(self, sorted_model):
        return self._finalize_model(sorted_model)
    
    def save_model(self, sorted_models, filename: str | dict[str, str] | None = None):
        return self._save_model(sorted_models, filename)
    
    def predict_on_test_set(self, sorted_models: dict | list, name: str) -> pd.DataFrame:
        return self._predict_on_test_set(sorted_models, name)


def main():
    label, training_output, budget_time, scaler, excel, kfold, outliers, \
    precision_weight, recall_weight, report_weight, difference_weight, strategy, best_model, \
    seed, drop, tune,  plot, optimize, selected = arg_parse()
    
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])
    training_output = Path(training_output)
    if len(outliers) == 1 and Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = [x.strip() for x in out.readlines()]
    
    feature = DataParser(label, excel)
    experiment = PycaretInterface("classification", feature.label, seed, budget_time=budget_time, best_model=best_model, 
                                  output_path=training_output)

    ranking_dict = dict(precision_weight=precision_weight, recall_weight=recall_weight,
                        difference_weight=difference_weight, report_weight=report_weight)
    training = Classifier(experiment, num_split, test_size, outliers, scaler, ranking_dict, drop, optimize=optimize,
                          selected=selected)
    
    sorted_results, sorted_models, top_params = training.run_training(feature, plot)

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