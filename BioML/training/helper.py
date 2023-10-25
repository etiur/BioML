from ..utilities import write_excel, scale
from pathlib import Path
from .base import Trainer, PycaretInterface
from collections import defaultdict
from typing import Callable, Iterable
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
import warnings


@dataclass
class DataParser:
    """
    A class for parsing feature and label data.

    Parameters
    ----------
    features : pd.DataFrame or str or list or np.ndarray
        The feature data.
    label : pd.Series or str or Iterable[int|float] or None, optional
        The label data. Defaults to None.
    outliers : dict[str, tuple], optional
        A dictionary containing the indices of the outliers in the feature and label data. Defaults to an empty dictionary.
    scaler : str, optional
        The type of scaler to use for feature scaling. Defaults to "robust".
    sheets : str or int, optional
        The sheet name or index to read from an Excel file. Defaults to 0.

    Attributes
    ----------
    features : pd.DataFrame
        The feature data.
    label : pd.Series or None
        The label data.
    outliers : dict[str, tuple]
        A dictionary containing the indices of the outliers in the feature and label data.
    scaler : str
        The type of scaler to use for feature scaling.
    sheets : str or int
        The sheet name or index to read from an Excel file.

    Methods
    -------
    read_features(features)
        Reads the feature data from a file or returns the input data.
    read_labels(label)
        Reads the label data from a file or returns the input data.
    scale(X_train, X_test)
        Scales the feature data.
    process(transformed_x, test_x, y_train, y_test)
        Concatenates the feature and label data so that it can be used by pycaret.
    """
    features: pd.DataFrame | str | list | np.ndarray
    label: pd.Series | str | Iterable[int|float] | None = None
    outliers: dict[str, tuple] = field(default_factory=lambda: defaultdict(tuple))
    scaler: str="robust"
    sheets: str | int = 0

    def __post_init__(self):
        self.features = self.read_features(self.features)
        if self.label:
            self.label = self.read_labels(self.label)

    def read_features(self, features: str | pd.DataFrame | list | np.ndarray) -> pd.DataFrame:
        """
        Reads the feature data from a file or returns the input data.

        Parameters
        ----------
        features : str or pd.DataFrame or list or np.ndarray
            The feature data.

        Returns
        -------
        pd.DataFrame
            The feature data as a pandas DataFrame.

        Raises
        ------
        TypeError
            If the input data type is not supported.
        """
        # concatenate features and labels
        if isinstance(features, str):
            if features.endswith(".csv"):
                return pd.read_csv(f"{features}", index_col=0) # the first column should contain the sample names

            elif features.endswith(".xlsx"):
                with pd.ExcelFile(features) as file:
                    if len(file.sheet_names) > 1:
                        warnings.warn(f"The excel file contains more than one sheet, only the sheet {self.sheets} will be used")

                return pd.read_excel(features, index_col=0, engine='openpyxl', sheet_name=self.sheets)
        
            
        elif isinstance(features, pd.DataFrame):
            return features
        elif isinstance(features, (list, np.ndarray)):
            return pd.DataFrame(features)

        self.log.error("features should be a csv or excel file, an array or a pandas DataFrame")
        raise TypeError("features should be a csv or excel file, an array or a pandas DataFrame")
        
    def read_labels(self, label: str | pd.Series) -> pd.Series:
        """
        Reads the label data from a file or returns the input data.

        Parameters
        ----------
        label : str or pd.Series
            The label data.

        Returns
        -------
        pd.Series
            The label data as a pandas Series.

        Raises
        ------
        TypeError
            If the input data type is not supported.
        """
        if isinstance(label, pd.Series):
            label.index.name = "target"
            return label
        
        elif isinstance(label, str):
            if Path(label).exists() and Path(label).suffix == ".csv":
                label = pd.read_csv(label, index_col=0)
                label.index.name = "target"
                return label    
            elif label in self.features.columns:
                lab = self.features[label]
                self.features.drop(label, axis=1, inplace=True)
                return lab
                    
        self.log.error("label should be a csv file, a pandas Series or inside features")
        raise TypeError("label should be a csv file, a pandas Series or inside features")
    
    def scale(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scales the train data and test data

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        X_test : pd.DataFrame
            The test data.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            The scaled training and test data.
        """
        #remove outliers
        X_train = X_train.loc[[x for x in X_train.index if x not in self.outliers["x_train"]]]
        X_test = X_test.loc[[x for x in X_test.index if x not in self.outliers["x_test"]]]

        transformed_x, scaler_dict, test_x = scale(self.scaler, X_train, X_test)

        return transformed_x, test_x
    
    def process(self, transformed_x: pd.DataFrame, test_x: pd.DataFrame, y_train: pd.Series, 
                y_test: pd.Series)-> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Concatenate the transformed feature data with the label data.

        Parameters
        ----------
        transformed_x : pd.DataFrame
            The transformed training feature data.
        test_x : pd.DataFrame
            The transformed testing feature data.
        y_train : pd.Series
            The training label data.
        y_test : pd.Series
            The testing label data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the concatenated training feature and label data, and the concatenated testing feature and label data.
        """
        transformed_x = pd.concat([transformed_x, y_train], axis=1) 
        test_x = pd.concat([test_x, y_test], axis=1)
        return transformed_x, test_x
    

def generate_training_results(model, training: Trainer, feature: DataParser, plot: tuple, 
                              tune: bool=False, strategy="holdout") -> dict[str, dict[str, tuple]]:
    """
    Generate training results for a given model, training object, and feature data.

    Parameters
    ----------
    model : Classifier or Regressor objects
        The model to use for training.
    training : Trainer
        The training object to use.
    feature : DataParser
        The feature data to use.
    plot : tuple
        A tuple containing the plot title and axis labels.
    tune : bool, optional
        Whether to tune the hyperparameters. Defaults to True.
    strategy : str, optional
        The strategy to use for training. Defaults to "holdout".

    Returns
    -------
    dict
        A dictionary containing the training results.
    """    
    sorted_results, sorted_models, top_params = model.run_training(training, feature, plot)

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
            

    return results

def evaluate_all_models(evaluation_fn: Callable, results: dict[str, dict[str, tuple]], training_output: str | Path) -> None:
    """
    Evaluate all models using the given evaluation function and save the results. The function used here plots the learning curve.
    It is easier to extend this function to evaluate it other ways.

    Parameters
    ----------
    evaluation_fn : Callable
        The evaluation function to use for evaluating the models.
    results : dict[str, dict[str, tuple]]
        A dictionary containing the results of the models to be evaluated.
    training_output : str | Path
        The path to the directory where the evaluation results will be saved.

    Returns
    -------
    None
        This function does not return anything, it only saves the evaluation results.
    """
    for tune_status, result_dict in results.items():
        for key, value in result_dict.items():
            if key == "stacked" or key == "majority":
                evaluation_fn(value[1], save=f"{training_output}/{tune_status}/{key}")
            elif tune_status == "tuned" and key == "holdout":
                for mod_name, model in value[1].items():
                    evaluation_fn(model, save=f"{training_output}/{tune_status}/{key}/{mod_name}")


def write_results(training_output: Path, sorted_results: pd.DataFrame, top_params: pd.Series | None = None, 
                  sheet_name: str|None=None) -> None:
    """
    Writes the training results and top hyperparameters to Excel files.

    Parameters
    ----------
    training_output : Path
        The path to the directory where the Excel files will be saved.
    sorted_results : pd.DataFrame
        A pandas DataFrame containing the sorted training results.
    top_params : pd.Series or None, optional
        A pandas Series containing the top hyperparameters. Defaults to None.
    sheet_name : str or None, optional
        The name of the sheet to write the results to. Defaults to None.

    Returns
    -------
    None
    """
    write_excel(training_output / "training_results.xlsx", sorted_results, sheet_name)
    if top_params is not None:
        write_excel(training_output / f"top_hyperparameters.xlsx", top_params, sheet_name)

def iterate_through_excel(model, excel_file: str | Path, feature: DataParser, training: Trainer) -> pd.DataFrame:
    with pd.ExcelFile(excel_file) as excel:
        for i, sheet in enumerate(excel.sheet_names):
            df = pd.read_excel(excel, sheet_name=sheet)
            feature

                

            results = generate_training_results(model, training, feature, ())
