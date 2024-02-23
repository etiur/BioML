from pathlib import Path
from typing import Callable, Iterable, Iterator
import pandas as pd
from dataclasses import dataclass
import json
import yaml
from typing import Any, Callable
from transformers import PreTrainedModel
import torch
from ..models import base
from .utils import write_excel


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
                raise ValueError(f"Unsupported file extension: {extension}")


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
                try:
                    evaluation_fn(value, save=f"{training_output}/evaluation_plots/{tune_status}/{key}")
                except AttributeError:
                    pass
            elif tune_status == "tuned" and key == "holdout":
                for mod_name, model in value.items(): # type: ignore
                    try:
                        evaluation_fn(model, save=f"{training_output}/evaluation_plots/{tune_status}/{key}/{mod_name}")
                    except AttributeError:
                        pass


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


def iterate_multiple_features(iterator: Iterator, parser:base.DataParser, label: str | list[int | float], 
                              training: base.Trainer, outliers: Iterable[str],
                              training_output: Path, **kwargs: Any) -> None:
    
    """
    Iterates over multiple input features and generates training results for each feature.

    Parameters
    ----------
    iterator : Iterator
        An iterator that yields a tuple of input features and sheet names.
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
        sorted_results, sorted_models, top_params = training.run_training(feature.feature, feature.label_name, **kwargs)
        index = sorted_results.index.unique(0)[:training.experiment.best_model]
        score = 0
        for i in index:
            score += training.arguments._calculate_score_dataframe(sorted_results.loc[i])
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


def estimate_model_size(model: PreTrainedModel, precision: torch.dtype):
    """
    Estimate the size of the model in memory.

    Parameters
    ----------
    model : PreTrainedModel
        The pre-trained model to estimate the size of.
    precision : torch.dtype
        The precision of the model's parameters. Can be torch.float16 for half precision or torch.float32 for single precision.

    Returns
    -------
    str
        The estimated size of the model in megabytes (MB), rounded to two decimal places.
    """
    num = 2 if precision==torch.float16 else 4
    size = round(model.num_parameters() * num/1000_000, 2)
    return f"{size} MB"


