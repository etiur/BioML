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
from collections import defaultdict, namedtuple
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, r2_score
from sklearn.metrics import classification_report as class_re
from helper import scale

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
    def __init__(self, selected_features: str | Path, label: str | Path,  ensemble_output: str | Path,
                 hyperparameter_path: str | Path, selected_sheets: list[str | int], outliers: list[str] | [str] =() ,
                 scaler: str ="robust",  num_splits: int =5, test_size: float=0.2):

        self.features = Path(selected_features)
        self.output_path = Path(ensemble_output)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.hyperparameter_path = Path(hyperparameter_path)
        self.selected_sheets = selected_sheets
        self.outliers = outliers
        self.scaler = scaler
        self.num_splits = num_splits
        self.test_size = test_size
        self.labels = pd.read_csv(label, index_col=0)

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

        return models

    def _check_features(self):
        with_split = True
        features = pd.read_excel(self.features, index_col=0, sheet_name=self.selected_sheets, header=[0, 1],
                                 engine='openpyxl')
        if not f"split_{0}" in list(features.values())[0].columns.unique(0):
            with_split = False
            features = pd.read_excel(self.features, index_col=0, sheet_name=self.selected_sheets, header=0,
                                     engine='openpyxl')
        return features, with_split
    def get_features(self, features, with_split):
        feature_indices = {}
        feature_dict = defaultdict(dict)
        for sheet, feature in features.items():
            skf = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=20)
            for ind, (train_index, test_index) in enumerate(skf.split(feature, self.labels)):
                feature_indices[ind] = (train_index, test_index)
                transformed_x, test_x, Y_test, Y_train = self._scale(feature, with_split, train_index, test_index, ind)
                feature_dict[sheet][ind] = (transformed_x, test_x, Y_test, Y_train)

        return feature_dict, feature_indices

    def predict(self, estimator, X_train, X_test):
        """
        train a model on the training set and make predictions on both the train and test sets
        """
        # Model predictions
        pred_test_y = estimator.predict(X_test)
        # training data prediction
        pred_train_y = estimator.predict(X_train)
        pred = namedtuple("prediction", ["pred_test_y", "pred_train_y"])
        return pred(*[pred_test_y, pred_train_y])

    def refit(self, feature_scaled, models):
        """ Fit the models and get the predictions"""
        results = defaultdict(dict)
        for sheet, feature_dict in feature_scaled.items():
            # refit the models with its corresponding index split and feature set
            for split_ind, feat in feature_dict.items():
                models[sheet][split_ind].fit(feat[0], feat[-1])
                name = models[sheet][split_ind].__class__.__name__
                pred = self.predict(models[sheet][split_ind], feat[0], feat[1])
                results[sheet][f"{name}_{split_ind}"] = {split_ind: pred}

        return results

    def get_score(self, pred, Y_train, Y_test):
        """ The function prints the scores of the models and the prediction performance """
        target_names = ["class 0", "class 1"]

        scalar_record = namedtuple("scores", ["train_mat", "test_matthews", "r2_train", "r2_test"])
        parameter_record = namedtuple("parameters", ["test_confusion", "tr_report", "te_report",
                                                     "train_confusion", "model_name"])
        # Model comparison
        # Training scores
        train_confusion = confusion_matrix(Y_train, pred.pred_train_y)
        tr_report = class_re(Y_train, pred.pred_train_y, target_names=target_names, output_dict=True)
        train_mat = matthews_corrcoef(Y_train, pred.pred_train_y)
        train_r2 = r2_score(Y_train, pred.pred_train_y)
        # Test metrics grid
        test_confusion = confusion_matrix(Y_test, pred.pred_test_y)
        test_matthews = matthews_corrcoef(Y_test, pred.pred_test_y)
        te_report = class_re(Y_test, pred.pred_test_y, target_names=target_names, output_dict=True)
        test_r2 = r2_score(Y_test, pred.pred_test_y)
        # save the results in namedtuples
        scalar_scores = scalar_record(*[train_mat, test_matthews, train_r2, test_r2])
        param_scores = parameter_record(*[test_confusion, tr_report, te_report, train_confusion,
                                          pred.fitted.__class__.__name__])

        return scalar_scores, param_scores

    def ensemble(self):
        """
        Use the model trained with its corresponding split set to predict on all split sets and see the result
        of the individual model as well as the ensemble
        """
        features, with_split = self._check_features()
        models = self.get_hyperparameter()
        feature_scaled, feature_indices = self.get_features(features, with_split)
        results = self.refit(feature_scaled, models)
        for sheet, feature in features.items():
            for ind in feature_indices.keys():
                # for the same ind I want to split each dataset within the same sheet set 5 times (25 times/ sheet)
                name = models[sheet][ind].__class__.__name__
                for i, (train_index, test_index) in enumerate(feature_indices.values()):
                    transformed_x, test_x, Y_test, Y_train = self._scale(feature, with_split, train_index,
                                                                             test_index, ind)
                    if ind == i: continue
                    # for each model I have the predictions for all the possible kfolds so that I can
                    pred = self.predict(models[sheet][ind], transformed_x, test_x)
                    results[sheet][f"{name}_{ind}"][i] = pred
        return results


    def _scale(self, features, with_split, train_index, test_index, split_index):
        # split and filter

        if with_split:
            feat_subset = features.loc[:, f"split_{split_index}"]
        else:
            feat_subset = features

        X_train = feat_subset.iloc[train_index]
        Y_train = feat_subset.iloc[train_index]
        X_test = feat_subset.iloc[test_index]
        Y_test = feat_subset.iloc[test_index]
        X_train = X_train.loc[[x for x in X_train.index if x not in self.outliers]]
        X_test = X_test.loc[[x for x in X_test.index if x not in self.outliers]]
        Y_train = Y_train.loc[[x for x in Y_train.index if x not in self.outliers]]
        Y_test = Y_test.loc[[x for x in Y_test.index if x not in self.outliers]]
        transformed_x, scaler_dict, test_x = scale(self.scaler, X_train, X_test)

        return transformed_x, test_x, Y_test, Y_train

    def to_dataframe(self, metric_scalar, parameter_list, split_index):
        matrix = namedtuple("confusion_matrix", ["true_n", "false_p", "false_n", "true_p"])
        # performance scores
        test_mathew = [x.test_matthews for x in metric_scalar]
        cv_score = [x.cv_score for x in metric_scalar]
        test_r2 = [x.r2_test for x in metric_scalar]
        train_mathew = [x.train_mat for x in metric_scalar]
        train_r2 = [x.r2_train for x in metric_scalar]
        # model parameters
        model_name = [x.model_name for x in parameter_list]
        # Taking the confusion matrix
        test_confusion = [matrix(*x.test_confusion.ravel()) for x in parameter_list]
        training_confusion = [matrix(*x.train_confusion.ravel()) for x in parameter_list]
        te_report = {num: pd.DataFrame(x.te_report).transpose() for num, x in zip(split_index, parameter_list)}
        te_report = pd.concat(te_report)
        te_report.index.names = ["split_index", "labels"]
        tr_report = {num: pd.DataFrame(x.tr_report).transpose() for num, x in zip(split_index, parameter_list)}
        tr_report = pd.concat(tr_report)
        tr_report.index.names = ["split_index", "labels"]
        report = pd.concat({"train":tr_report.T, "test": te_report.T})
        # Separating confusion matrix into individual elements
        test_true_n = [y.true_n for y in test_confusion]
        test_false_p = [y.false_p for y in test_confusion]
        test_false_n = [y.false_n for y in test_confusion]
        test_true_p = [y.true_p for y in test_confusion]

        training_true_n = [z.true_n for z in training_confusion]
        training_false_p = [z.false_p for z in training_confusion]
        training_false_n = [z.false_n for z in training_confusion]
        training_true_p = [z.true_p for z in training_confusion]

        dataframe = pd.DataFrame([split_index, test_true_n, test_true_p, test_false_p, test_false_n, training_true_n,
                                  training_true_p, training_false_p, training_false_n, cv_score,
                                  train_mathew, test_mathew, train_r2, test_r2, model_name])

        dataframe = dataframe.transpose()
        dataframe.columns = ["split_index", "test_tn", "test_tp", "test_fp", "test_fn", "train_tn", "train_tp",
                             "train_fp", "train_fn", "CV_MCC", "train_MCC", "test_MCC", "train_R2", "test_R2",
                             "model_name"]
        dataframe.set_index("split_index", inplace=True)

        return dataframe, report.T

    def ranking(self):
        """
        TODO Again rank the results based on the performance metrics
        """
        pass

    def run(self):
        """
        TODO Run all previous functions
        """
        results = self.ensemble()
