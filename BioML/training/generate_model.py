from typing import Iterable
from .base import PycaretInterface,  DataParser
from pathlib import Path
import argparse
import pandas as pd
from collections import defaultdict
import joblib
from ..utilities import write_excel
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, ShuffleSplit
from .classification import Classifier
from .regression import Regressor


def arg_parse():
    parser = argparse.ArgumentParser(description="Generate the models from the ensemble")

    parser.add_argument("-i", "--training_features", required=True,
                        help="The file to where the features for the training are in excel or csv format")
    parser.add_argument("-sc", "--scaler", default="robust", choices=("robust", "standard", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to RobustScaler")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format")
    parser.add_argument("-o", "--model_output", required=False,
                        help="The directory for the generated models",
                        default="models")
    parser.add_argument("-ot", "--outliers", nargs="+", required=False, default=(),
                        help="A list of outliers if any, the name should be the same as in the excel file with the "
                             "filtered features, you can also specify the path to a file in plain text format, each "
                             "record should be in a new line")
    parser.add_argument("-s", "--selected_models", nargs="+", required=True, 
                        help="The models to use, can be regression or classification")
    parser.add_argument("-p", "--problem", required=False, 
                        default="classification", choices=("classification", "regression"), help="The problem type")
    parser.add_argument("-op", "--optimize", required=False, 
                        default="MCC", choices=("MCC", "Prec.", "Recall", "F1", "AUC", "Accuracy", "Average Precision Score", 
                                                "RMSE", "R2", "MSE", "MAE", "RMSLE", "MAPE"), 
                        help="The metric to optimize")
    parser.add_argument("-st", "--strategy", required=False, choices=("holdout", "kfold"), default="holdout",
                        help="The spliting strategy to use")
    parser.add_argument("-m", "--model_strategy", help=f"The strategy to use for the model, choices are majority, stacking or simple:[model_index], model index should be an integer", default="majority")

    args = parser.parse_args()

    return [args.training_features, args.label, args.scaler, args.model_output,
            args.outliers, args.selected_models, args.problem, args.optimize, args.strategy, 
            args.model_strategy]


class GenerateModel:
    def __init__(self, trainer: Regressor | Classifier, selected_models: tuple[str] | str | dict[str, tuple[str, ...]],
                 model_output="models", problem="classification", optimize="MCC"):
        
        self.trainer = trainer
        self.model_output = Path(model_output)
        self.selected_models = selected_models
        self.problem = problem
        self.optimize = optimize
    
    def run_generate(self, feature: DataParser):
        """
        A function that splits the data into training and test sets and then trains the models
        using cross-validation but only on the training data

        Parameters
        ----------
        feature : DataParser
            A class containing the training samples and the features

        Returns
        -------
        pd.DataFrame
            A dictionary with the sorted results from pycaret
        list[models]
            A dictionary with the sorted models from pycaret
        pd.DataFrame
         
        """
        self.log.info("------ Running holdout -----")
        
        sorted_results, sorted_models, top_params = self.trainer.run_training(feature, plot=())
        return sorted_models


    def finalize_model(self, sorted_model, index: int | dict[str, int] | None = None):
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
            
    def save_model(self, sorted_models, filename: str | dict[str, int] | None=None, 
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


def main():
    excel, label, scaler, hyperparameter_path, model_output, num_thread, outliers, sheets = arg_parse()
    if Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = [x.strip() for x in out.readlines()]
    generate = GenerateModel(excel, hyperparameter_path, label, sheets, scaler, num_thread, outliers,
                             model_output)
    generate.refit_save()


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()

