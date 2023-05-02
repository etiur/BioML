from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from pathlib import Path
import numpy as np
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description="Detect outliers from the selected features")

    parser.add_argument("-e", "--excel", required=True,
                        help="The file to where the selected features are saved in excel format",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-o", "--ensemble_output", required=True,
                        help="The path to the output for the ensemble results",
                        default="ensemble_results")
    parser.add_argument("-hy", "--hyperparameters", required=True,
                        help="Path to the hyperparameters file")
    parser.add_argument("-s", "--sheets", required=False,nargs="+",
                        help="Names or index of the selected sheets for both features and hyperparameters")

    args = parser.parse_args()

    return [args.excel, args.ensemble_output, args.hyperparameters, args.sheets]


def interesting_classifiers(name, **kwargs):
    """
    All classifiers
    """
    classifiers = {
        "random_tree": RandomForestClassifier(**kwargs),
        "extra_tree": ExtraTreeClassifier(**kwargs),
        "sgd": SGDClassifier(**kwargs),
        "ridge": RidgeClassifier(**kwargs),
        "passive": PassiveAggressiveClassifier(**kwargs),
        "mlp": MLPClassifier(**kwargs),
        "svc":SVC(**kwargs),
        "xgboost":XGBClassifier(**kwargs),
        "light": LGBMClassifier(**kwargs),
        "knn": KNeighborsClassifier(**kwargs)
    }

    return classifiers[name]


class Ensemble:
    def __init__(self, selected_features, ensemble_output, hyperparameters, selected_sheets):
        self.selected_features = Path(selected_features)
        self.output_path = Path(ensemble_output)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.hyperparameters = Path(hyperparameters)
        self.selected_models = selected_sheets

    @staticmethod
    def vote(val=1, *args):
        """
        Hard voting for the ensembles

        Parameters
        ___________
        args: list[arrays]
            A list of prediction arrays
        """
        vote_ = []
        index = []
        mean = np.mean(args, axis=0)
        for s, x in enumerate(mean):
            if x == 1 or x == 0:
                vote_.append(int(x))
            elif x >= val:
                vote_.append(1)
                index.append(s)
            elif x < val:
                vote_.append(0)
                index.append(s)

        return vote_, index

    def get_hyperparameter(self):
        """
        TODO A function that will read the hyperparameters from the selected sheets and return a dictionary with
        TODO the models
        """
        pass

    def refit(self):
        """
        TODO Fit the dictionary of models with its corresponding split index data and return the fitted models
        """
        pass

    def ensemble(self):
        """
        TODO Use the model trained with its corresponding split set to predict on all split sets and see the result
        TODO of the individual model as well as the ensemble
        """
        pass

    def ranking(self):
        """
        TODO Again rank the results based on the performance metrics
        """
        pass

    def run(self):
        """
        TODO Run all previous functions
        """
        pass