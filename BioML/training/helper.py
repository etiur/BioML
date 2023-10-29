from ..utilities import write_excel
from pathlib import Path
from .base import Trainer
from collections import defaultdict
from typing import Callable, Iterable, Iterator
import pandas as pd
from dataclasses import dataclass
import numpy as np
import warnings
import json
import yaml
from typing import Any, Callable
from ..custom_errors import NotSupportedError


@dataclass(slots=True)
class DataParser:
    """
    A class for parsing feature and label data.

    Parameters
    ----------
    features : pd.DataFrame or str or list or np.ndarray
        The feature data.
    label : pd.Series or str or Iterable[int|float] or None, optional
        The label data. Defaults to None.
    outliers : Iterable[str], optional
        An iterable containing the indices of the outliers in the feature and label data. Defaults to an empty ().
    sheets : str or int, optional
        The sheet name or index to read from an Excel file. Defaults to 0.

    Attributes
    ----------
    features : pd.DataFrame
        The feature data.
    label : pd.Series or None
        The label data.
    outliers : Iterable[str]
        An iterable containing the indices of the outliers in the feature and label data.
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
    outliers: Iterable[str] = ()
    sheets: str | int |None = None

    def __post_init__(self):
        self.features = self.read_features(self.features)
        if self.label:
            self.label = self.read_labels(self.label)
            if not isinstance(self.label, str):
                self.features = pd.concat([self.features, self.label], axis=1)
                self.label = self.label.index.name

        self.features = self.remove_outliers(self.features, self.outliers)

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
        NotSupportedError
            If the input data type is not supported.
        """
        # concatenate features and labels
        match features:
            case str(feature) if feature.endswith(".csv"):
                return pd.read_csv(f"{features}", index_col=0) # the first column should contain the sample names
            case str(feature) if feature.endswith(".xlsx"):
                sheets = self.sheets if self.sheets else 0
                with pd.ExcelFile(features) as file:
                    if len(file.sheet_names) > 1:
                        warnings.warn(f"The excel file contains more than one sheet, only the sheet {sheets} will be used")
                return pd.read_excel(features, index_col=0, engine='openpyxl', sheet_name=sheets)
            case pd.DataFrame() as feature:
                return feature
            case list() | np.ndarray() as feature:
                return pd.DataFrame(feature)
            case _:
                raise NotSupportedError("features should be a csv or excel file, an array or a pandas DataFrame")
        
    def read_labels(self, label: str | pd.Series) -> str | pd.Series:
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
        match label:
            case pd.Series() as labels:
                labels.index.name = "target"
                return labels
            case str(labels) if Path(labels).exists() and Path(labels).suffix == ".csv":
                labels = pd.read_csv(labels, index_col=0)
                labels.index.name = "target"
                return labels
            case str(labels) if labels in self.features.columns:
                return labels
            case list() | np.ndarray() as labels:
                return pd.Series(labels, index=self.features.index, columns=["target"])
            case _:
                raise NotSupportedError(f"label should be a csv file, an array, a pandas Series or inside features: you provided {label}")
    
    def remove_outliers(self, training_features: pd.DataFrame, outliers: Iterable[str]):
        """
        Remove outliers from the train data

        Parameters
        ----------
        training_features : pd.DataFrame
            The training data.

        outliers: Iterable[str]
            An iterable containing the indices to remove from the training set
            
        Returns
        -------
        pd.DataFrame
            The data with outliers removed.
        """
        #remove outliers
        training_features.loc[[x for x in training_features.index if x not in outliers]]

        return training_features
    
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
            A tuple containing the concatenated training feature and label data, 
            and the concatenated testing feature and label data.
        """
        transformed_x = pd.concat([transformed_x, y_train], axis=1) 
        test_x = pd.concat([test_x, y_test], axis=1)
        return transformed_x, test_x
    

@dataclass(slots=True)
class FileParser:
    file_path: str | Path

    def load(self, extension: str="json") -> dict[str, Any]:
        with open(self.file_path) as file:
            if extension == "json":
                return json.load(file)
            elif extension == "yaml":
                return yaml.load(file, Loader=yaml.FullLoader)
            else:
                raise NotSupportedError(f"Unsupported file extension: {extension}")


def generate_training_results(model, training: Trainer, feature: DataParser, plot: tuple, 
                              tune: bool=False, **kwargs) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Generate training results for a given model, training object, and feature data.

    Parameters
    ----------
    model : classification.Classifier or regression.Regressor objects
        The model to use for training.
    training : Trainer
        The training object to use.
    feature : DataParser
        The feature data to use.
    plot : tuple
        A tuple containing the plot title and axis labels.
    tune : bool, optional
        Whether to tune the hyperparameters. Defaults to True.
    **kwargs : dict
        Additional keyword arguments to pass to the pycaret setup function.

    Returns
    -------
    tuple[dict, dict]
        A tuple of dictionary containing the training results and the models.
    """    
    sorted_results, sorted_models, top_params = model.run_training(training, feature, plot, **kwargs)
    if 'dummy' in sorted_results.index.unique(0)[:3]:
        warnings.warn(f"Dummy model is in the top {list(sorted_results.index.unique(0)).index('dummy')} models, turning off tuning")
        tune = False

    # saving the results in a dictionary and writing it into excel files
    models_dict = defaultdict(dict)
    results = defaultdict(dict)
    # save the results
    results["not_tuned"]["holdout"] = sorted_results, top_params
    stacked_results, stacked_models, stacked_params = training.stack_models(sorted_models)
    results["not_tuned"]["stacked"] = stacked_results, stacked_params
    majority_results, majority_models = training.create_majority_model(sorted_models)
    results["not_tuned"]["majority"] = majority_results
    #satev the models
    models_dict["not_tuned"]["holdout"] = sorted_models
    models_dict["not_tuned"]["stacked"] = stacked_models
    models_dict["not_tuned"]["majority"] = majority_models

    if tune:
        # save the results
        sorted_result_tune, sorted_models_tune, top_params_tune = training.retune_best_models(sorted_models)
        results["tuned"]["holdout"] = sorted_result_tune, top_params_tune
        stacked_results_tune, stacked_models_tune, stacked_params_tune = training.stack_models(sorted_models_tune)
        results["tuned"]["stacked"] = stacked_results_tune, stacked_params_tune
        majority_results_tune, majority_models_tune = training.create_majority_model(sorted_models_tune)
        results["tuned"]["majority"] = majority_results_tune   

        #save the models
        models_dict["tuned"]["holdout"] = sorted_models_tune
        models_dict["tuned"]["stacked"] = stacked_models_tune
        models_dict["tuned"]["majority"] = majority_models_tune

    return results, models_dict


def evaluate_all_models(evaluation_fn: Callable, results: dict[str, dict[str, tuple | dict]], 
                        training_output: str | Path) -> None:
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
                evaluation_fn(value, save=f"{training_output}/evaluation_plots/{tune_status}/{key}")
            elif tune_status == "tuned" and key == "holdout":
                for mod_name, model in value.items():
                    evaluation_fn(model, save=f"{training_output}/evaluation_plots/{tune_status}/{key}/{mod_name}")


def generate_test_prediction(models_dict: dict[str, dict], training: Trainer, sorting_function: Callable) -> dict[str, pd.DataFrame]:
    """
    Generate test set predictions for a given set of models.

    Parameters
    ----------
    models_dict : dict[str, dict]
        A dictionary containing the models to use for each tuning status.
    training : Trainer
        An instance of the Trainer class used to train the models.
    sorting_function : Callable
        A function used to sort the predictions.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the prediction results for each tuning status.
    """
    prediction_results = {}
    for tune_status, result_dict in models_dict.items():
        predictions = []
        for key, models in result_dict.items():
            # get the test set prediction results
            predictions.append(training.predict_on_test_set(models))
        prediction_results[tune_status] = sorting_function(pd.concat(predictions))
    return prediction_results


def write_results(training_output: Path | str, sorted_results: pd.DataFrame, top_params: pd.Series | None = None, 
                  sheet_name: str|None=None) -> None:
    """
    Writes the training results and top hyperparameters to Excel files.

    Parameters
    ----------
    training_output : Path | str
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
    training_output = Path(training_output)
    training_output.mkdir(exist_ok=True, parents=True)
    write_excel(training_output / "training_results.xlsx", sorted_results, sheet_name)
    if top_params is not None:
        write_excel(training_output / f"top_hyperparameters.xlsx", top_params, sheet_name)


def sort_regression_prediction(dataframe: pd.DataFrame, optimize: str="RSME") -> pd.DataFrame:
    """
    Sorts the predictions of a regression model based on a specified optimization metric and R2 score.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the predictions of the regression model.
    optimize : str, optional
        The name of the optimization metric to use for sorting the predictions. Default is "RMSE".

    Returns
    -------
    pd.DataFrame
        The sorted DataFrame of predictions.
    """
    if optimize == "R2":
        return dataframe.sort_values(optimize, ascending=False)
    if optimize != "R2":
        return dataframe.sort_values(optimize,ascending=True)
    

def sort_classification_prediction(dataframe: pd.DataFrame, optimize:str="MCC", prec_weight:float=1.2, 
                                   recall_weight:float=0.8, report_weight:float=0.6) -> pd.DataFrame:
    """
    Sorts the predictions of a classification model based on a specified optimization metric and precision/recall scores.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the predictions of the classification model.
    optimize : str, optional
        The name of the optimization metric to use for sorting the predictions. Default is "MCC".
    prec_weight : float, optional
        The weight to give to the precision score when sorting the predictions. Default is 1.2.
    recall_weight : float, optional
        The weight to give to the recall score when sorting the predictions. Default is 0.8.
    report_weight : float, optional
        The weight to give to the classification report scores when sorting the predictions. Default is 0.6.

    Returns
    -------
    pd.DataFrame
        The sorted DataFrame of predictions.
    """
    sort = dataframe.loc[(dataframe[optimize] + report_weight * (prec_weight * dataframe["Prec."] + 
                    recall_weight * dataframe["Recall"])).sort_values(ascending=False).index]
    return sort


def iterate_multiple_features(iterator: Iterator, model, label: str | list[int | float], 
                              training: Trainer, outliers: Iterable[str],
                              training_output: Path, **kwargs) -> None:
    
    """
    Iterates over multiple input features and generates training results for each feature.

    Parameters
    ----------
    iterator : Iterator
        An iterator that yields a tuple of input features and sheet names.
    model : Any
        The machine learning model to use for training.
    label : str or list[int or float]
        The label or list of labels to use for training.
    training : Trainer
        The training object to use for training the model.
    outliers : Iterable[str]
        An iterable containing the names of the outlier detection methods to use for each sheet.
    training_output : Path
        The path to the directory where the training results will be saved.

    Returns
    -------
    None
    """

    performance_list = []
    for input_feature, sheet in iterator:
        feature = DataParser(input_feature, label=label, sheets=sheet, outliers=outliers)
        sorted_results, sorted_models, top_params = model.run_training(training, feature, plot=(), **kwargs)
        index = sorted_results.index.unique(0)[:training.experiment.best_model]
        score = 0
        for i in index:
            score += model._calculate_score_dataframe(sorted_results.loc[i])
        performance_list.append((sheet, sorted_results.loc[index], score))
    performance_list.sort(key=lambda x: x[2], reverse=True)
    for sheet, performance, score in performance_list:
        write_results(training_output, performance, sheet_name=sheet)
    

def iterate_excel(excel_file: str | Path):
    """
    Iterates over the sheets of an Excel file and yields a tuple of the sheet data and sheet name.

    Parameters
    ----------
    excel_file : str or Path
        The path to the Excel file.

    Yields
    ------
    Tuple[pd.DataFrame, str]
        A tuple of the sheet data and sheet name.
    """
    with pd.ExcelFile(excel_file) as file:
        for sheet in file.sheet_names:
            df = pd.read_excel(excel_file, index_col=0, sheet_name=sheet)
            yield df, sheet



