from BioML.utilities import interesting_classifiers
from pathlib import Path
import argparse
import pandas as pd
from collections import defaultdict
import joblib
from BioML.utilities import scale, modify_param


def arg_parse():
    parser = argparse.ArgumentParser(description="Generate the models from the ensemble")

    parser.add_argument("-e", "--excel", required=False,
                        help="The file to where the selected features are saved in excel format",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-hp", "--hyperparameter_path", required=False, help="Path to the hyperparameter file",
                        default="training_results/hyperparameters.xlsx")
    parser.add_argument("-n", "--num_thread", required=False, default=10, type=int,
                        help="The number of threads to use for the parallelization of outlier detection")
    parser.add_argument("-sc", "--scaler", default="robust", choices=("robust", "standard", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to RobustScaler")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format")
    parser.add_argument("-o", "--model_output", required=False,
                        help="The directory for the generated models",
                        default="models")
    parser.add_argument("-s", "--sheets", required=False, nargs="+",
                        help="Names or index of the selected sheets for both features and hyperparameters and the "
                             "index of the models in this format-> sheet (name, index):index model1,index model2 "
                             "without the spaces. If only index or name of the sheets, it is assumed that all kfold models "
                             "are selected. It is possible to have kfold indices in one sheet and in another ones "
                             "without")
    parser.add_argument("-ot", "--outliers", nargs="+", required=False, default=(),
                        help="A list of outliers if any, the name should be the same as in the excel file with the "
                             "filtered features, you can also specify the path to a file in plain text format, each "
                             "record should be in a new line")

    args = parser.parse_args()

    return [args.excel, args.label, args.scaler, args.hyperparameter_path, args.model_output, args.num_thread,
            args.outliers, args.sheets]


class GenerateModel:
    def __init__(self, selected_features, hyperparameter_path, label, sheets, scaler="robust", num_threads=10,
                 outliers=(), model_output="models"):
        self.selected_features = selected_features
        self.hyperparameter_path = hyperparameter_path
        self.label = pd.read_csv(label, index_col=0)
        self.scaler = scaler
        self.num_threads = num_threads
        self.outliers = outliers
        self.model_output = Path(model_output)
        self.selected_sheets = []
        self.selected_kfolds = {}
        self._check_sheets(sheets)
        self.with_split = True

    def _check_sheets(self, sheets):
        for x in sheets:
            indices = x.split(":")
            sh = indices[0]
            if sh.isdigit():
                sh = int(sh)
            self.selected_sheets.append(sh)
            if len(indices) > 1:
                kfold = tuple(int(x) for x in indices[1].split(","))
                self.selected_kfolds[sh] = kfold
            else:
                self.selected_kfolds[sh] = ()

    def get_hyperparameter(self):
        """
        A function that will read the hyperparameters and features from the selected sheets and return a dictionary with
        the models and features
        """
        models = defaultdict(dict)
        model_indices = defaultdict(list)
        hp = pd.read_excel(self.hyperparameter_path, sheet_name=self.selected_sheets, index_col=[0, 1, 2],
                           engine='openpyxl')
        hp = {key: value.where(value.notnull(), None) for key, value in hp.items()}
        for sheet, data in hp.items():
            for ind in data.index.unique(level=0):
                if self.selected_kfolds[sheet] and ind not in self.selected_kfolds[sheet]: continue
                name = data.loc[ind].index.unique(level=0)[0]
                param = data.loc[(ind, name), 0].to_dict()
                param = modify_param(param, name, self.num_threads)
                # each sheet should have 5 models representing each split index
                models[sheet][ind] = interesting_classifiers(name, param)
                model_indices[sheet].append(ind)

        return models, model_indices

    def _check_features(self):
        features = pd.read_excel(self.selected_features, index_col=0, header=[0, 1], engine='openpyxl')
        if f"split_{0}" not in features.columns.unique(0):
            self.with_split = False
        if self.with_split:
            features = pd.read_excel(self.selected_features, index_col=0, sheet_name=self.selected_sheets,
                                     header=[0, 1], engine='openpyxl')
        else:
            features = pd.read_excel(self.selected_features, index_col=0, sheet_name=self.selected_sheets, header=0,
                                     engine='openpyxl')
        return features

    def get_features(self, features, model_index):

        feature_dict = defaultdict(dict)
        label_dict =defaultdict(dict)
        for sheet, feature in features.items():
            random_state = 22342
            for ind in model_index[sheet]:
                if self.with_split:
                    sub_feat = feature.loc[:, f"split_{ind}"].sample(frac=1, random_state=random_state)
                else:
                    sub_feat = feature.sample(frac=1, random_state=random_state)
                transformed, scaler_dict = scale(self.scaler, sub_feat)
                feature_dict[sheet][ind] = transformed
                label_dict[sheet][ind] = self.label.sample(frac=1, random_state=random_state).values.ravel()
                random_state += 10000

        return feature_dict, label_dict

    def refit_save(self):
        """Fit the models and get the predictions"""
        models, model_indices = self.get_hyperparameter()
        features = self._check_features()
        feature_dict, label_dict = self.get_features(features, model_indices)
        for sheet, shuffled_feature in feature_dict.items():
            out = self.model_output / f"sheet_{sheet}"
            out.mkdir(parents=True, exist_ok=True)
            # refit the models with its corresponding index split and feature set
            for split_ind, feat in shuffled_feature.items():
                label = label_dict[sheet][split_ind]
                models[sheet][split_ind].fit(feat, label)
                name = models[sheet][split_ind].__class__.__name__
                joblib.dump(models[sheet][split_ind], out / f"{name}_{split_ind}.joblib")


def main():
    excel, label, scaler, hyperparameter_path, model_output, num_thread, outliers, sheets = arg_parse()
    if Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = [x.strip() for x in out.readlines()]
    generate = GenerateModel(excel, hyperparameter_path, label, sheets, scaler, num_thread, outliers,
                             model_output)
    generate.refit_save()


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()

