from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from pathlib import Path
import numpy as np
import argparse
import pandas as pd
from collections import defaultdict

def arg_parse():
    parser = argparse.ArgumentParser(description="Detect outliers from the selected features")

    parser.add_argument("-e", "--excel", required=True,
                        help="The file to where the selected features are saved in excel format",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-o", "--ensemble_output", required=True,
                        help="The path to the output for the ensemble results",
                        default="ensemble_results")
    parser.add_argument("-hp", "--hyperparameter_path", required=True,
                        help="Path to the hyperparameter file")
    parser.add_argument("-s", "--sheets", required=False,nargs="+",
                        help="Names or index of the selected sheets for both features and hyperparameters")

    args = parser.parse_args()

    return [args.excel, args.ensemble_output, args.hyperparameter_path, args.sheets]


def interesting_classifiers(name, params):
    """
    All classifiers
    """
    classifiers = {
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "SGDClassifier": SGDClassifier,
        "RidgeClassifier": RidgeClassifier,
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
        "MLPClassifier": MLPClassifier,
        "SVC":SVC,
        "XGBClassifier":XGBClassifier,
        "LGBMClassifier": LGBMClassifier,
        "KNeighborsClassifier": KNeighborsClassifier
    }

    return classifiers[name](**params)


class Ensemble:
    def __init__(self, selected_features: str | Path, ensemble_output: str | Path, hyperparameter_path: str | Path,
                 selected_sheets: list[str | int]):
        self.features = Path(selected_features)
        self.output_path = Path(ensemble_output)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.hyperparameter_path = Path(hyperparameter_path)
        self.selected_sheets = selected_sheets

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
        A function that will read the hyperparameters and features from the selected sheets and return a dictionary with
        the models and features
        """
        selected_features = defaultdict(dict)
        models = defaultdict(dict)
        hp = pd.read_excel(self.hyperparameter_path, sheet_name=self.selected_sheets, index_col=[0, 1, 2],
                           engine='openpyxl')
        hp = {key: value.where(value.notnull(), None) for key, value in hp.items()}
        for sheet, data in hp.items():
            for ind in data.index.unique(level=0):
                name = data.loc[ind].index.unique(level=0)[0]
                param = data.loc[(ind, name), 0].to_dict()
                # each sheet should have 5 models representing each split index
                models[sheet][ind] = interesting_classifiers(name, param)

        features = pd.read_excel(self.features, index_col=0, sheet_name=self.selected_sheets, header=[0, 1],
                                 engine='openpyxl')
        for sheet, data in features.items():
            for ind in data.columns.unique(level=0):
                feat = data.loc[:, ind]
                # each sheet should have 5 feature sets representing each split index
                selected_features[sheet][int(ind.split("_")[1])] = feat
        return selected_features, models


    def refit(self):
        """
        TODO Fit the dictionary of models with its corresponding features and split index    and return the fitted models
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