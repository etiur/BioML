from .base import DataParser, PycaretInterface, Trainer
from pathlib import Path
import argparse
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
    parser.add_argument("-se", "--selected_models", nargs="+", required=True, 
                        help="The models to use, can be regression or classification")
    parser.add_argument("-p", "--problem", required=False, 
                        default="classification", choices=("classification", "regression"), help="The problem type")
    parser.add_argument("-op", "--optimize", required=False, 
                        default="MCC", choices=("MCC", "Prec.", "Recall", "F1", "AUC", "Accuracy", "Average Precision Score", 
                                                "RMSE", "R2", "MSE", "MAE", "RMSLE", "MAPE"), 
                        help="The metric to optimize")
    parser.add_argument("-st", "--strategy", required=False, choices=("holdout", "kfold"), default="holdout",
                        help="The spliting strategy to use")
    parser.add_argument("-m", "--model_strategy", help=f"The strategy to use for the model, choices are majority, stacking or simple:model_index, model index should be an integer", default="majority")
    parser.add_argument("--seed", required=True, help="The seed for the random state")
    parser.add_argument("-k", "--kfold_parameters", required=False,
                        help="The parameters for the kfold in num_split:test_size format", default="5:0.2")
    
    parser.add_argument("--tune", action="store_false", required=False, help="If to tune the best models")
    
    args = parser.parse_args()

    return [args.training_features, args.label, args.scaler, args.model_output,
            args.outliers, args.selected_models, args.problem, args.optimize, args.strategy, 
            args.model_strategy, args.seed, args.kfold_parameters, args.tune]


class GenerateModel:

    def finalize_model(self, sorted_model, index: int | None = None):
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
            
    def save_model(self, model, filename: str):
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
    
    def train_by_strategy(sorted_models: dict, model_strategy: str, training: Trainer, optimize: str):
        if "majority" in model_strategy:
            models = training.create_majority_model(sorted_models, optimize)
            index = None
        elif "stacking" in model_strategy:
            models = training.stack_models(sorted_models, optimize)
            index = None
        elif "simple" in model_strategy:
            models = sorted_models
            index = int(model_strategy.split(":")[1])
    
        return models, index
    

def main():
    training_features, label, scaler, model_output, outliers, selected_models, \
    problem, optimize, strategy, model_strategy, seed, kfold, tune = arg_parse()
    if outliers and Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = [x.strip() for x in out.readlines()]
    outliers = {"x_train": outliers, "x_test": outliers}
    
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])

    # instantiate everything to run training
    feature = DataParser(training_features, label, outliers=outliers, scaler=scaler)
    experiment = PycaretInterface(problem, feature.label, seed, best_model=len(selected_models))
    training = Trainer(experiment, num_split, test_size)
    if problem == "classification":
        model = Classifier(drop=None, selected=selected_models)
    elif problem == "regression":
        model = Regressor(drop=None, selected=selected_models)
    sorted_results, sorted_models, top_params = model.run_training(training, feature, plot=())
    if tune:
        sorted_results, sorted_models, top_params = training.retune_best_models(sorted_models, optimize)
    # generate the final model
    generate = GenerateModel()
    models, index =  generate.train_by_strategy(sorted_models, model_strategy, training, optimize)
    final_model = generate.finalize_model(models, index)
    generate.save_model(final_model, model_output)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()

