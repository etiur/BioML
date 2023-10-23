from typing import Iterable
from .base import PycaretInterface, Trainer, DataParser, write_results
from pathlib import Path
import argparse
import pandas as pd
from collections import defaultdict
import joblib
from ..utilities import write_excel
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, ShuffleSplit


def arg_parse():
    parser = argparse.ArgumentParser(description="Generate the models from the ensemble")

    parser.add_argument("-e", "--excel", required=False,
                        help="The file to where the selected features are saved in excel format",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-hp", "--hyperparameter_path", required=False, help="Path to the hyperparameter file",
                        default="training_results/hyperparameters.xlsx")
    parser.add_argument("-n", "--num_thread", required=False, default=10, type=int,
                        help="The number of threads to use for the parallelization of outlier detection")
    parser.add_argument("-sc", "--scaler", default="robust", choices=("robust", "standard", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to RobustScaler")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format")
    parser.add_argument("-o", "--model_output", required=False,
                        help="The directory for the generated models",
                        default="models")
    parser.add_argument("-s", "--sheets", required=False, nargs="+",
                        help="Names or index of the selected sheets for both features and hyperparameters and the "
                             "index of the models in this format-> sheet (name, index):index model1,index model2 "
                             "without the spaces. If only index or name of the sheets, it is assumed that all kfold models "
                             "are selected. It is possible to have kfold indices in one sheet and in another ones "
                             "without")
    parser.add_argument("-ot", "--outliers", nargs="+", required=False, default=(),
                        help="A list of outliers if any, the name should be the same as in the excel file with the "
                             "filtered features, you can also specify the path to a file in plain text format, each "
                             "record should be in a new line")

    args = parser.parse_args()

    return [args.excel, args.label, args.scaler, args.hyperparameter_path, args.model_output, args.num_thread,
            args.outliers, args.sheets]


class GenerateModel(Trainer):
    def __init__(self, model: PycaretInterface, selected_models: tuple[str] | str | dict[str, tuple[str, ...]], num_splits=5, 
                 test_size=0.2, outliers: tuple[str, ...]=(), scaler="robust", model_output="models",
                 problem="classification"):
        super().__init__(model, num_splits=num_splits, test_size=test_size, outliers=outliers, 
                         scaler=scaler)

        self.model_output = Path(model_output)
        self.selected_models = selected_models
        self.problem = problem
    
    def run_holdout(self, feature: DataParser):
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
        if self.problem == "classification":
            X_train, X_test = train_test_split(feature.features, test_size=self.test_size, random_state=self.experiment.seed, 
                                           stratify=feature.features[feature.label])
        elif self.problem == "regression":
            X_train, X_test = train_test_split(feature.features, test_size=self.test_size, random_state=self.experiment.seed)
        
        sorted_results, sorted_models, top_params = self.setup_holdout(X_train, X_test, self._calculate_score_dataframe, plot=False, selected=self.selected_models)
        return sorted_models

    def run_kfold(self, feature: DataParser):
        """
        A function that splits the data into kfolds of training and test sets and then trains the models
        using cross-validation but only on the training data. It is a nested cross-validation

        Parameters
        ----------
        feature : pd.DataFrame
            A dataframe containing the training samples and the features
 

        Returns
        -------
        tuple[pd.DataFrame, dict[str, models], pd.Series]
            A dictionary with the sorted results and sorted models from pycaret organized by split index or kfold index
        """
        self.log.info("------ Running kfold -----")
        if self.problem == "classification":
            skf = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.experiment.seed)
        elif self.problem == "regression":
            skf = ShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.experiment.seed)

        sorted_results, sorted_models, top_params = self.setup_kfold(feature.features, feature.label, skf, plot=False, with_split=feature.with_split,
                                                                     selected=self.selected_models)
        return sorted_models

    def stack_models(self, sorted_models: dict, optimize="MCC", probability_theshold: float | None = None, meta_model=None):
        return self._stack_models(sorted_models, optimize, probability_theshold, meta_model)
    
    def create_majority_model(self, sorted_models: dict, optimize: str = "MCC", probability_theshold: float | None = None, weights: Iterable[float] | None = None):
        return self._create_majority_model(sorted_models, optimize, probability_theshold, weights)

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

