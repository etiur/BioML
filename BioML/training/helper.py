from ..utilities import write_excel
from pathlib import Path
from . import base
from collections import defaultdict
from typing import Callable, Iterable, Iterator
import pandas as pd
from dataclasses import dataclass
import numpy as np
import warnings
import json
import yaml
from typing import Any, Callable, Protocol
from ..custom_errors import NotSupportedDataError


class Modelor(Protocol):
    drop: Iterable[str]
    selected: Iterable[str]
    optimize: str

    def _calculate_score_dataframe(self, dataframe: pd.DataFrame) -> int | float:
        ...
    
    def run_training(self, trainer: base.Trainer, feature: base.DataParser, plot: tuple[str, ...], 
                     **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
        ...


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
                raise NotSupportedDataError(f"Unsupported file extension: {extension}")


def generate_training_results(model: Modelor, training: base.Trainer, feature: base.DataParser, plot: tuple, 
                              tune: bool=False, **kwargs: Any) -> tuple[dict[str, dict], dict[str, dict]]:
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
    results["not_tuned"]["majority"] = majority_results, 
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
        results["tuned"]["majority"] = majority_results_tune,   

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
                for mod_name, model in value.items(): # type: ignore
                    evaluation_fn(model, save=f"{training_output}/evaluation_plots/{tune_status}/{key}/{mod_name}")


def generate_test_prediction(models_dict: dict[str, dict], training: base.Trainer, 
                             sorting_function: Callable) -> dict[str, pd.DataFrame]:
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
    write_excel(training_output / "training_results.xlsx", sorted_results, sheet_name) # type: ignore
    if top_params is not None:
        write_excel(training_output / f"top_hyperparameters.xlsx", top_params, sheet_name) # type: ignore


def sort_regression_prediction(dataframe: pd.DataFrame, optimize: str="RSME") -> pd.DataFrame: # type: ignore
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


def iterate_multiple_features(iterator: Iterator, model: Modelor, parser:base.DataParser, label: str | list[int | float], 
                              training: base.Trainer, outliers: Iterable[str],
                              training_output: Path, **kwargs: Any) -> None:
    
    """
    Iterates over multiple input features and generates training results for each feature.

    Parameters
    ----------
    iterator : Iterator
        An iterator that yields a tuple of input features and sheet names.
    model : Any
        The machine learning model to use for training.
    parser : DataParser
        The data parser to use for parsing the input features.
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
        feature = parser(input_feature, label=label, sheets=sheet, outliers=outliers)
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



