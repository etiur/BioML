from pathlib import Path
import numpy as np
import argparse
from collections import defaultdict, namedtuple
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, r2_score
from sklearn.metrics import classification_report as class_re
from utilities import scale, write_excel, interesting_classifiers, modify_param
from itertools import combinations


def arg_parse():
    parser = argparse.ArgumentParser(description="Detect outliers from the selected features")

    parser.add_argument("-e", "--excel", required=False,
                        help="The file to where the selected features are saved in excel format",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-o", "--ensemble_output", required=False,
                        help="The path to the output for the ensemble results",
                        default="ensemble_results")
    parser.add_argument("-hp", "--hyperparameter_path", required=False, help="Path to the hyperparameter file",
                        default="training_features/hyperparameters.xlsx")
    parser.add_argument("-s", "--sheets", required=True, nargs="+",
                        help="Names or index of the selected sheets for both features and hyperparameters")
    parser.add_argument("-va", "--prediction_threshold", required=False, default=1.0, type=float,
                        help="Between 0.5 and 1 and determines what considers to be a positive prediction, if 1 only"
                             "those predictions where all models agrees are considered to be positive")
    parser.add_argument("-n", "--num_thread", required=False, default=10, type=int,
                        help="The number of threads to use for the parallelization of outlier detection")
    parser.add_argument("-pw", "--precision_weight", required=False, default=1, type=float,
                        help="Weights to specify how relevant is the precision for the ranking of the different features")
    parser.add_argument("-rw", "--recall_weight", required=False, default=0.8, type=float,
                        help="Weights to specify how relevant is the recall for the ranking of the different features")
    parser.add_argument("-c0", "--class0_weight", required=False, default=0.5, type=float,
                        help="Weights to specify how relevant is the f1, precision and recall scores of the class 0"
                             " or the negative class for the ranking of the different features with respect to class 1 or "
                             "the positive class")
    parser.add_argument("-rpw", "--report_weight", required=False, default=0.25, type=float,
                        help="Weights to specify how relevant is the f1, precision and recall for the ranking of the "
                             "different features with respect to MCC and the R2 which are more general measures of "
                             "the performance of a model")
    parser.add_argument("-dw", "--difference_weight", required=False, default=0.8, type=float,
                        help="How important is to have similar training and test metrics")
    parser.add_argument("-k", "--kfold_parameters", required=False,
                        help="The parameters for the kfold in num_split:test_size format", default="5:0.2")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format")
    parser.add_argument("-sc", "--scaler", required=False, default="robust", choices=("robust", "standard", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to RobustScaler")
    parser.add_argument("-ot", "--outliers", nargs="+", required=False, default=(),
                        help="A list of outliers if any, the name should be the same as in the excel file with the "
                             "filtered features, you can also specify the path to a file in plain text format, each "
                             "record should be in a new line")

    args = parser.parse_args()

    return [args.excel, args.label, args.scaler, args.ensemble_output, args.hyperparameter_path, args.sheets,
            args.prediction_threshold, args.kfold_parameters, args.outliers, args.precision_weight,
            args.recall_weight, args.class0_weight, args.report_weight, args.difference_weight, args.num_thread]


class EnsembleClassification:
    def __init__(self, label: str | Path, selected_sheets: list[str | int],
                 selected_features: str | Path = "training_features/selected_features.xlsx",
                 ensemble_output: str | Path = "ensemble_results",
                 hyperparameter_path: str | Path = "training_features/hyperparameters.xlsx",  outliers: list[str] = (),
                 scaler: str = "robust",  num_splits: int = 5, test_size: float = 0.2,
                 prediction_threshold: float = 1.0, precision_weight=1, recall_weight=1, report_weight=0.4,
                 difference_weight=0.8, class0_weight=0.5, num_threads=10):

        self.features = Path(selected_features)
        self.output_path = Path(ensemble_output)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.hyperparameter_path = Path(hyperparameter_path)
        self.selected_sheets = selected_sheets
        self.single_feature = False
        if len(self.selected_sheets) == 1:
            self.single_feature = True
        self.outliers = outliers
        self.scaler = scaler
        self.num_splits = num_splits
        self.test_size = test_size
        self.labels = pd.read_csv(label, index_col=0)
        self.prediction_threshold = prediction_threshold
        self.pre_weight = precision_weight
        self.rec_weight = recall_weight
        self.report_weight = report_weight
        self.difference_weight = difference_weight
        self.class0_weight = class0_weight
        self.num_threads = num_threads


    def vote(self, *args):
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
            elif x >= self.prediction_threshold:
                vote_.append(1)
                index.append(s)
            elif x < self.prediction_threshold:
                vote_.append(0)
                index.append(s)

        return vote_, index

    def _scale(self, features, with_split, train_index, test_index, split_index):
        # split and filter

        if with_split:
            feat_subset = features.loc[:, f"split_{split_index}"]
        else:
            feat_subset = features

        X_train = feat_subset.iloc[train_index]
        Y_train = self.labels.iloc[train_index]
        X_test = feat_subset.iloc[test_index]
        Y_test = self.labels.iloc[test_index]
        X_train = X_train.loc[[x for x in X_train.index if x not in self.outliers]]
        X_test = X_test.loc[[x for x in X_test.index if x not in self.outliers]]
        Y_train = Y_train.loc[[x for x in Y_train.index if x not in self.outliers]].values.ravel()
        Y_test = Y_test.loc[[x for x in Y_test.index if x not in self.outliers]].values.ravel()
        transformed_x, scaler_dict, test_x = scale(self.scaler, X_train, X_test)

        return transformed_x, test_x, Y_test, Y_train

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
                param = modify_param(param, name, self.num_threads)
                # each sheet should have 5 models representing each split index
                models[sheet][ind] = interesting_classifiers(name, param)

        return models

    def _check_features(self):
        with_split = True
        features = pd.read_excel(self.features, index_col=0, sheet_name=self.selected_sheets, header=[0, 1],
                                 engine='openpyxl')
        if f"split_{0}" not in list(features.values())[0].columns.unique(0):
            with_split = False
            features = pd.read_excel(self.features, index_col=0, sheet_name=self.selected_sheets, header=0,
                                     engine='openpyxl')
        return features, with_split

    def get_features(self, features, with_split):
        feature_indices = {}
        feature_dict = defaultdict(dict)
        label_dict = {}
        for sheet, feature in features.items():
            skf = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=20)
            for ind, (train_index, test_index) in enumerate(skf.split(feature, self.labels)):
                feature_indices[ind] = (train_index, test_index)
                transformed_x, test_x, Y_test, Y_train = self._scale(feature, with_split, train_index, test_index, ind)
                feature_dict[sheet][ind] = (transformed_x, test_x)
                if ind not in label_dict:
                    label_dict[ind] = (Y_train, Y_test)

        return feature_dict, feature_indices, label_dict

    @staticmethod
    def predict(estimator, X_train, X_test):
        """
        train a model on the training set and make predictions on both the train and test sets
        """
        # Model predictions
        pred_test_y = estimator.predict(X_test)
        # training data prediction
        pred_train_y = estimator.predict(X_train)
        pred = namedtuple("prediction", ["pred_test_y", "pred_train_y"])
        return pred(*[pred_test_y, pred_train_y])

    def refit(self, feature_scaled, models, label_dict):
        """ Fit the models and get the predictions"""
        results = defaultdict(dict)
        for sheet, feature_dict in feature_scaled.items():
            # refit the models with its corresponding index split and feature set
            for split_ind, feat in feature_dict.items():
                label = label_dict[split_ind]
                models[sheet][split_ind].fit(feat[0], label[0])
                name = models[sheet][split_ind].__class__.__name__
                pred = self.predict(models[sheet][split_ind], feat[0], feat[1])
                if sheet not in results[split_ind]:
                    results[split_ind][sheet] = {}
                results[split_ind][sheet][f"{name}_{split_ind}"] = pred

        return results

    def predict_all_split_sets(self, features, with_split, feature_indices, models, results):
        """
        Use the model trained with its corresponding split set to predict on all split sets and see the result
        of the individual models as well as the ensemble.
        Note the ensemble results within the same sheet is not very informative since the models are predicting on
        samples that they were trained with but if the results after aggregating the predictions do not improve you
        might want to discard that model and reduce the complexity of the ensemble.
        The ensemble between different sheets however should be more useful.
        """
        for sheet, feature in features.items():
            for ind in feature_indices.keys():
                # for the same ind I want to split each dataset within the same sheet set 5 times (25 times/ sheet)
                name = models[sheet][ind].__class__.__name__
                for kfold, (train_index, test_index) in enumerate(feature_indices.values()):
                    transformed_x, test_x, Y_test, Y_train = self._scale(feature, with_split, train_index,
                                                                         test_index, ind)
                    if ind == kfold: continue
                    # for each model I have the predictions for all the possible kfolds
                    pred = self.predict(models[sheet][ind], transformed_x, test_x)
                    results[kfold][sheet][f"{name}_{ind}"] = pred

        return results

    @staticmethod
    def _transform_results(results):
        different_sheet_same_index = defaultdict(dict)
        for kfold, model_dict in results.items():
            for sheet, pred_dict in model_dict.items():
                for model_name, pred in pred_dict.items():
                    if model_name.split("_")[-1] == str(kfold):
                        # 5 kfold and num_sheet predictions per kfold
                        different_sheet_same_index[kfold][f"{sheet}_{model_name}"] = pred

        return different_sheet_same_index

    def analyse_vote(self, predictions, label, name):
        train_pred, train_indices = self.vote(*tuple(x.pred_train_y for x in predictions))
        test_pred, test_indices = self.vote(*tuple(x.pred_test_y for x in predictions))
        scalar_scores, param_scores = self.get_score(train_pred, test_pred, label[0], label[1])
        dataframe, report = self.to_dataframe(scalar_scores, param_scores, name)
        return dataframe, report

    def ensemble_voting(self, results, label_dict):
        ensemble_results = defaultdict(list)
        # first I get the results from the ensembling of different sheets
        different_sheet_same_index = self._transform_results(results)
        # return the results from the original model to see if they are the same and then ensemble if there are more
        # than 1 sheet
        for kfold, pred_dict in different_sheet_same_index.items():
            label = label_dict[kfold]
            model_names, predictions = pred_dict.keys(), pred_dict.values()
            if not self.single_feature:
                ensemble_results[kfold].append(self.analyse_vote(predictions, label, "_".join(model_names)))
            for i, (model_name, prediction) in enumerate(zip(model_names, predictions)):
                ensemble_results[kfold].append(self.analyse_vote((prediction,), label, model_name))

            for sheet, pred_dict in results[kfold].items():
                model_names = pred_dict.keys()
                for n in range(2, len(model_names)+1):
                    for comb in combinations(model_names, n):
                        sub_pred = [pred_dict[x] for x in comb]
                        ensemble_results[kfold].append(self.analyse_vote(sub_pred, label, "_".join(comb)))

        # save the dataframe and the reports
        ensemble_results = {kfold: tuple(zip(*sorted(value, key=self.rank_results, reverse=True)))
                            for kfold, value in ensemble_results.items()}

        return ensemble_results

    def run(self):
        """
        Each split index will be an Excel sheet, I will ensemble the results per sheet and save the results
        """
        features, with_split = self._check_features()
        models = self.get_hyperparameter()
        feature_scaled, feature_indices, label_dict = self.get_features(features, with_split)
        results = self.refit(feature_scaled, models, label_dict)
        results = self.predict_all_split_sets(features, with_split, feature_indices, models, results)
        ensemble_results = self.ensemble_voting(results, label_dict)
        # save the results
        with (pd.ExcelWriter(self.output_path / "ensemble_report.xlsx", mode="w", engine="openpyxl") as writer1,
        pd.ExcelWriter(self.output_path / "ensemble_results.xlsx", mode="w", engine="openpyxl") as writer2):
            for kfold, res in ensemble_results.items():
                dataframe = pd.concat(res[0])
                report = pd.concat({x.index.name: x for x  in res[1]})
                write_excel(writer1, report, f"split_{kfold}")
                write_excel(writer2, dataframe, f"split_{kfold}")

    @staticmethod
    def get_score(pred_train_y, pred_test_y, Y_train, Y_test):
        """ The function prints the scores of the models and the prediction performance """
        target_names = ["class 0", "class 1"]

        scalar_record = namedtuple("scores", ["train_mat", "test_matthews", "r2_train", "r2_test"])
        parameter_record = namedtuple("parameters", ["test_confusion", "tr_report", "te_report",
                                                     "train_confusion"])
        # Model comparison
        # Training scores
        train_confusion = confusion_matrix(Y_train, pred_train_y)
        tr_report = class_re(Y_train, pred_train_y, target_names=target_names, output_dict=True)
        train_mat = matthews_corrcoef(Y_train, pred_train_y)
        train_r2 = r2_score(Y_train, pred_train_y)
        # Test metrics grid
        test_confusion = confusion_matrix(Y_test, pred_test_y)
        test_matthews = matthews_corrcoef(Y_test, pred_test_y)
        te_report = class_re(Y_test, pred_test_y, target_names=target_names, output_dict=True)
        test_r2 = r2_score(Y_test, pred_test_y)
        # save the results in namedtuples
        scalar_scores = scalar_record(*[train_mat, test_matthews, train_r2, test_r2])
        param_scores = parameter_record(*[test_confusion, tr_report, te_report, train_confusion])

        return scalar_scores, param_scores

    @staticmethod
    def to_dataframe(scalar_scores, param_scores, name):
        """ A function that transforms the data into dataframes"""
        matrix = namedtuple("confusion_matrix", ["true_n", "false_p", "false_n", "true_p"])
        # Taking the confusion matrix
        test_confusion = matrix(*param_scores.test_confusion.ravel())
        training_confusion = matrix(*param_scores.train_confusion.ravel())
        # Separating confusion matrix into individual elements
        test_true_n = test_confusion.true_n
        test_false_p = test_confusion.false_p
        test_false_n = test_confusion.false_n
        test_true_p = test_confusion.true_p
        training_true_n = training_confusion.true_n
        training_false_p = training_confusion.false_p
        training_false_n = training_confusion.false_n
        training_true_p = training_confusion.true_p
        # coonstructing the dataframe
        dataframe = pd.DataFrame([test_true_n, test_true_p, test_false_p, test_false_n, training_true_n,
                                  training_true_p, training_false_p, training_false_n, scalar_scores.train_mat,
                                  scalar_scores.test_matthews, scalar_scores.r2_train, scalar_scores.r2_test])
        dataframe = dataframe.transpose()
        dataframe.columns = ["test_tn", "test_tp", "test_fp", "test_fn", "train_tn", "train_tp",
                             "train_fp", "train_fn", "train_MCC", "test_MCC", "train_R2", "test_R2"]
        dataframe.index = [name]
        te_report = pd.DataFrame(param_scores.te_report)
        tr_report = pd.DataFrame(param_scores.tr_report)
        report = pd.concat({"train": tr_report, "test": te_report})
        report.index.name = name
        return dataframe, report.T

    def _calculate_score_dataframe(self, dataframe):
        return ((dataframe["train_MCC"] + dataframe["test_MCC"] + dataframe["train_R2"] + dataframe["test_R2"])
                - self.difference_weight * (abs(dataframe["test_MCC"] - dataframe["train_MCC"]) +
                                            abs(dataframe["test_R2"] - dataframe["train_R2"]))).sum()

    def _calculate_score_report(self, report, class_label):
        class_level = report.loc[report.index.get_level_values(0) == class_label,
                                 report.columns.get_level_values(1).isin(["precision", "recall"])]
        pre_sum = self.pre_weight * (class_level.loc[class_label, ("test", "precision")] +
                                     class_level.loc[class_label, ("train", "precision")])
        rec_sum = self.rec_weight * (class_level.loc[class_label, ("test", "recall")] +
                                     class_level.loc[class_label, ("train", "recall")])
        pre_diff = self.pre_weight * abs(class_level.loc[class_label, ("test", "precision")] -
                                         class_level.loc[class_label, ("train", "precision")])
        rec_diff = self.rec_weight * abs(class_level.loc[class_label, ("test", "recall")] -
                                         class_level.loc[class_label, ("train", "recall")])
        return (pre_sum + rec_sum - self.difference_weight * (pre_diff + rec_diff))

    def rank_results(self, result_item: list) -> float:

        dataframe, report = result_item
        score_dataframe = self._calculate_score_dataframe(dataframe)
        score_report_class0 = self._calculate_score_report(report, "class 0")
        score_report_class1 = self._calculate_score_report(report, "class 1")
        # Calculate the overall rank as the sum of the scores -> the greater, the better
        rank = score_dataframe + (self.report_weight * (self.class0_weight * score_report_class0 +
                                                        score_report_class1)/2)

        return rank

def main():
    selected_features, label, scaler, ensemble_output, hyperparameter_path, sheets, prediction_threshold, \
    kfold_parameters, outliers, precision_weight, recall_weight, class0_weight, report_weight, \
    difference_weight, num_thread = arg_parse()
    num_split, test_size = int(kfold_parameters.split(":")[0]), float(kfold_parameters.split(":")[1])
    if Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = tuple(x.strip() for x in out.readlines())
    if sheets[0].isdigit():
        sheets = [int(x) for x in sheets]
    ensemble = EnsembleClassification(selected_features, label,  ensemble_output, hyperparameter_path, sheets, outliers,
                 scaler,  num_split, test_size, prediction_threshold, precision_weight, recall_weight, report_weight,
                 difference_weight, class0_weight, num_thread)
    ensemble.run()