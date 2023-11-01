from .base import PycaretInterface, Trainer, DataParser
from pathlib import Path
import argparse
from .classification import Classifier
from .regression import Regressor
from .helper import FileParser
from typing import Any



def arg_parse():
    parser = argparse.ArgumentParser(description="Generate the models from the ensemble")

    parser.add_argument("-i", "--training_features", required=True,
                        help="The file to where the features for the training are in excel or csv format")
    parser.add_argument("-sc", "--scaler", default="robust", choices=("robust", "zscore", "minmax"),
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
    parser.add_argument("-se", "--selected_models", nargs="+", required=True, 
                        help="The models to use, can be regression or classification")
    parser.add_argument("-p", "--problem", required=False, 
                        default="classification", choices=("classification", "regression"), help="The problem type")
    parser.add_argument("-op", "--optimize", required=False, 
                        default="MCC", choices=("MCC", "Prec.", "Recall", "F1", "AUC", "Accuracy", "Average Precision Score", 
                                                "RMSE", "R2", "MSE", "MAE", "RMSLE", "MAPE"), 
                        help="The metric to optimize")
    parser.add_argument("-m", "--model_strategy", help=f"The strategy to use for the model, choices are majority, stacking or simple:model_index, model index should be an integer", default="majority")
    parser.add_argument("--seed", required=True, help="The seed for the random state")
    parser.add_argument("-k", "--kfold_parameters", required=False,
                        help="The parameters for the kfold in num_split:test_size format", default="5:0.2")
    parser.add_argument("-sh", "--sheet_name", required=False, default=None, 
                        help="The sheet name for the excel file if the training features is in excel format")
    parser.add_argument("--tune", action="store_false", required=False, help="If to tune the best models")
    parser.add_argument("-j", "--setup_config", required=False, default=None,
                        help="A json or yaml file for the setup_configurations")
    parser.add_argument("-ni", "--num_iter", default=30, type=int, required=False, 
                        help="The number of iterations for the hyperparameter search")
    args = parser.parse_args()

    return [args.training_features, args.label, args.scaler, args.model_output, args.outliers, args.selected_models, 
            args.problem, args.optimize, args.model_strategy, args.seed, args.kfold_parameters, args.tune, args.sheet_name, 
            args.setup_config, args.num_iter]


class GenerateModel:
    """
    A class that generates machine learning models based on a specified strategy.

    Attributes
    ----------
    trainer : Trainer
        The Trainer object containing the data and parameters for training the models.
        
    """

    def  __init__(self, trainer: Trainer):
        self.trainer = trainer

    def finalize_model(self, sorted_model: list[Any] | dict[str, Any] | Any, 
                       index: int | None = None):
        """
        Finalize the model by training it with all the data including the test set.

        Parameters
        ----------
        sorted_model : Any
            The model or models to finalize.
        index : int, optional
            The index of the model to finalize, by default None.

        Returns
        -------
        Any
            The finalized models

        """
        match sorted_model:
            case [*list_models]:
                mod = list_models[index]
                return self.trainer.experiment.finalize_model(mod)
            
            case {**dict_models}: # for holdout models, it should be sorted
                final = {}
                mod = list(dict_models.values())[index]
                return self.trainer.experiment.finalize_model(mod)
            
            case model: # for stacked or majority models
                final = self.trainer.experiment.finalize_model(model)
                return final
            
    def save_model(self, model: Any, filename: str):
        """
        Save the model

        Parameters
        ----------
        model : Any
            The trained model.
        filename : str, dict[str, str]
            The name of the file to save the model.
        """
        model_output = Path(filename)
        model_output.parent.mkdir(exist_ok=True, parents=True)
        if model_output.suffix:
            model_output = model_output.with_suffix("")
        self.trainer.experiment.save(model, str(model_output))
    
    def train_by_strategy(self, sorted_models: dict, model_strategy: str):
        """
        Trains a machine learning model based on a specified strategy.

        Parameters
        ----------
        sorted_models : dict
            A dictionary containing the sorted models to use for training.
        model_strategy : str
            The name of the strategy to use for training. Can be "majority", "stacking", or "simple:index".

        Returns
        -------
        Any
            the trained models

        Examples
        --------
        >>> sorted_models = {"model_1": model_1, "model_2": model_2, "model_3": model_3}
        >>> trained_models, index = train_by_strategy(sorted_models, "majority")
        ... # Trains a majority model using the sorted models.
        >>> trained_models, index = train_by_strategy(sorted_models, "stacking")
        ... # Trains a stacked model using the sorted models.
        >>> trained_models, index = train_by_strategy(sorted_models, "simple:1")
        ... # Trains the second model in the sorted models.
        """

        if "majority" in model_strategy:
            _, models = self.trainer.create_majority_model(sorted_models)
        elif "stacking" in model_strategy:
            _, models, _ = self.trainer.stack_models(sorted_models)
        elif "simple" in model_strategy:
            index = int(model_strategy.split(":")[1])
            models = list(sorted_models.values())[index]
    
        return models
    

def main():
    training_features, label, scaler, model_output, outliers, selected_models, \
    problem, optimize, model_strategy, seed, kfold, tune, sheet, setup_config, num_iter = arg_parse()
    if outliers and Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = tuple(x.strip() for x in out.readlines())
    if setup_config:
        file = FileParser(setup_config)
        setup_config = file.load(extension=setup_config.split(".")[-1])

    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])

    # instantiate everything to run training
    feature = DataParser(training_features, label, outliers=outliers, sheets=sheet)
    
    experiment = PycaretInterface(problem, feature.label, seed, scaler=scaler,best_model=len(selected_models), optimize=optimize, 
                                  output_path=model_output)
    training = Trainer(experiment, num_split, num_iter)
    if problem == "classification":
        model = Classifier(drop=(), selected=selected_models, test_size=test_size, optimize=optimize)
    elif problem == "regression":
        model = Regressor(drop=(), selected=selected_models, test_size=test_size, optimize=optimize)
    _, sorted_models, _ = model.run_training(training, feature, plot=(), **setup_config)
    if tune:
        _, sorted_models, _ = training.retune_best_models(sorted_models)
    # generate the final model
    generate = GenerateModel(training)
    models =  generate.train_by_strategy(sorted_models, model_strategy)
    final_model = generate.finalize_model(models)
    generate.save_model(final_model, model_output)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()

