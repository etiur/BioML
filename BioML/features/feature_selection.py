import argparse
import pandas as pd
from BioML.utilities import scale, analyse_composition, write_excel
from pathlib import Path
from ITMO_FS.filters.univariate import select_k_best, UnivariateFilter
from ITMO_FS.filters.unsupervised import TraceRatioLaplacian
# Multiprocess instead of Multiprocessing solves the pickle problem in Windows (might be different in linux)
# but it has its own errors. Use multiprocessing.get_context('fork') seems to solve the problem but only available
# in Unix. Altough now it seems that with fork the programme can hang indefinitely so use spaw instead
# https://medium.com/devopss-hole/python-multiprocessing-pickle-issue-e2d35ccf96a9
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from ITMO_FS.filters.multivariate import STIR, TraceRatioFisher
from collections import defaultdict
import shap
import numpy as np
import matplotlib.pyplot as plt
import random
from multiprocessing import get_context  # https://pythonspeed.com/articles/python-multiprocessing/
import time
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


def arg_parse():
    parser = argparse.ArgumentParser(description="Preprocess and Select the best features, only use it if the feature came from possum or ifeatures")

    parser.add_argument("-f", "--features", required=False,
                        help="The path to the training features that contains both ifeature and possum in csv format",
                        default="training_features/every_features.csv")
    parser.add_argument("-l", "--label", required=True,
                        help="The path to the labels of the training set in a csv format if not in the features,"
                             "if present in the features csv use the flag to specify the label column name")
    parser.add_argument("-r", "--feature_range", required=False, default="none:none:none",
                        help="Specify the minimum and maximum of number of features in start:stop:step format or "
                             "a single integer. Start will default to num samples / 10, Stop will default to num samples / 2 and step will be (stop - step / 5)")
    parser.add_argument("-n", "--num_thread", required=False, default=10, type=int,
                        help="The number of threads to use for parallelizing the feature selection")
    parser.add_argument("-v", "--variance_threshold", required=False, default=0, type=float,
                        help="The variance the feature has to have, 0 means that the comlun has the same value for all samples. None to deactivate")
    parser.add_argument("-s", "--scaler", required=False, default="robust", choices=("robust", "zscore", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to RobustScaler")
    parser.add_argument("-e", "--excel_file", required=False,
                        help="The file path to where the selected features will be saved in excel format",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-k", "--kfold_parameters", required=False,
                        help="The parameters for the kfold in num_split:test_size format", default="5:0.2")
    parser.add_argument("-rt", "--rfe_steps", required=False, type=int,
                        help="The number of steps for the RFE algorithm, the more step the more precise "
                             "but also more time consuming and might lead to overfitting", default=30)
    parser.add_argument("-p", "--plot", required=False, action="store_false",
                        help="Default to true, plot the feature importance using shap")
    parser.add_argument("-pk", "--plot_num_features", required=False, default=20, type=int,
                        help="How many features to include in the plot")
    parser.add_argument("-nf", "--num_filters", required=False, type=int,
                        help="The number univariate filters to use maximum 10", default=10)
    parser.add_argument("-se", "--seed", required=False, type=int, default=None,
                        help="The seed number used for reproducibility")
    parser.add_argument("-st", "--strategy", required=False, choices=("holdout", "kfold"), default="holdout",
                        help="The spliting strategy to use")

    args = parser.parse_args()

    return [args.features, args.label, args.variance_threshold, args.feature_range, args.num_thread, args.scaler,
            args.excel_file, args.kfold_parameters, args.rfe_steps, args.plot, args.plot_num_features, args.num_filters,
            args.seed, args.strategy]


class FeatureSelection:
    def __init__(self, label, excel_file, features="training_features/every_features.csv", variance_thres=0,
                 num_thread=10, scaler="robust", num_split=5, test_size=0.2, num_filters=10, seed=None):
        """
        _summary_

        Parameters
        ----------
        label : str
            Path to the label or the name of the column with the label if included in feature file
        excel_file : str
            excel file where the selected features will be saved
        features : str, optional
            The features extracted for the training, by default "training_features/every_features.csv"
        variance_thres : int, optional
            The maximum number of repeated values for a column, if > the column will be eliminated, by default 7
        num_thread : int, optional
            Cpus for the parallelization of the selection, by default 10
        scaler : str, optional
            The name for the scaler. robust for RobustScaler, minmax for MinMaxScaler and zscore for StadardScaler, by default "robust"
        num_split : int, optional
            Number of kfold splits, by default 5
        test_size : float, optional
            The size of the test set, by default 0.2
        num_filters : int, optional
            The number of feature selection algorithms to use, by default 10
        """
        print("Reading the features")
        self.features = pd.read_csv(f"{features}", index_col=0) # the first column shoudl contain the sample names
        analyse_composition(self.features)
        if Path(label).exists():
            self.label = pd.read_csv(label, index_col=0)
        else:
            self.label = self.features[label]
            self.features.drop(label, axis=1, inplace=True)
            
        self._check_label(label) 
        self.variance_thres = variance_thres
        self.num_thread = num_thread
        self.scaler = scaler
        self.num_splits = num_split
        self.test_size = test_size
        self.excel_file = Path(excel_file)
        self.num_filters = num_filters
        if not str(self.excel_file).endswith(".xlsx"):
            self.excel_file = self.excel_file.with_suffix(".xlsx")
        self.excel_file.parent.mkdir(parents=True, exist_ok=True)
        if seed:
            self.seed = seed
        else:
            self.seed = int(time.time())
        print("seed:", self.seed)

    def _check_label(self, label):
        if len(self.label) != len(self.features):
            try:
                self.label = self.label.loc[self.features.index]
                label_path = Path(label)
                if not label_path.with_stem("labels_wrong").exists():
                    label_path.rename(label_path.with_stem("labels_wrong"))
                self.label.to_csv(label)
            except KeyError:
                raise KeyError("several names, keys or sequence ids in the fasta file are not present in the label")

    def preprocess(self):
        """
        Eliminate low variance features
        """
        if self.variance_thres is not None:
            variance = VarianceThreshold(self.variance_thres)
            fit = variance.fit_transform(self.features)
            self.features = pd.DataFrame(fit, index=self.features.index, 
                                         columns=variance.get_feature_names_out())

        return self.features

    @staticmethod
    def univariate(X_train, Y_train, num_features, feature_names, filter_name):
        """Features are considered one at the time and we are using statistical filters"""
        ufilter = UnivariateFilter(filter_name, select_k_best(num_features))
        ufilter.fit(X_train, Y_train)
        scores = {x: v for x, v in zip(feature_names, ufilter.feature_scores_)}
        # sorting the features
        if filter_name != "LaplacianScore":
            scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
        else:
            scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1])))
        return scores

    @staticmethod
    def multivariate(X_train, Y_train, num_features, feature_names, filter_name):
        if filter_name == "STIR":
            ufilter = STIR(num_features, k=5).fit(X_train, Y_train)
            scores = {x: v for x, v in zip(feature_names, ufilter.feature_scores_)}
        else:
            ufilter = TraceRatioFisher(num_features).fit(X_train, Y_train)
            scores = {x: v for x, v in zip(feature_names, ufilter.score_)}
        scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
        return scores

    @staticmethod
    def unsupervised(X_train, num_features, feature_names, filter_name):
        if "Trace" in filter_name:
            ufilter = TraceRatioLaplacian(num_features).fit(X_train)
            scores = {x: v for x, v in zip(feature_names, ufilter.score_)}

        scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))

        return scores

    def random_classification(self, X_train, Y_train, feature_names):
        rfc_classi = rfc(class_weight="balanced_subsample", random_state=27, max_features=0.5, max_samples=0.8,
                         min_samples_leaf=5, n_estimators=200, n_jobs=self.num_thread)
        rfc_classi.fit(X_train, Y_train)
        gini_importance = pd.Series(rfc_classi.feature_importances_, index=feature_names)
        gini_importance.sort_values(ascending=False, inplace=True)

        return gini_importance

    def xgbtree(self, X_train, Y_train, feature_names, split_ind, plot=True, plot_num_features=20):
        """computes the feature importance"""
        XGBOOST = xgb.XGBClassifier(learning_rate=0.01, n_estimators=200, max_depth=4, gamma=0,
                                    subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                    nthread=self.num_thread, seed=27)

        XGBOOST.fit(X_train, Y_train)
        important_features = XGBOOST.get_booster().get_score(importance_type='gain')
        important_features = pd.Series(important_features)
        important_features.sort_values(ascending=False, inplace=True)
        xgboost_explainer = shap.TreeExplainer(XGBOOST, X_train, feature_names=feature_names)
        shap_values = xgboost_explainer.shap_values(X_train, Y_train)
        shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), feature_names).sort_values(ascending=False)
        shap_importance = shap_importance.loc[lambda x: x > 0]

        if plot:
            shap_dir = (self.excel_file.parent / "shap_features")
            shap_dir.mkdir(parents=True, exist_ok=True)
            shap_importance.to_csv(shap_dir / f"shap_importance_kfold{split_ind}.csv")
            shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type='bar', show=False,
                              max_display=plot_num_features)
            plt.savefig(shap_dir / f"shap_kfold{split_ind}_top_{plot_num_features}_features.png", dpi=800)
            shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False,
                              max_display=plot_num_features)
            plt.savefig(shap_dir / f"feature_influence_on_model_prediction_kfold{split_ind}.png", dpi=800)
        return shap_importance

    @staticmethod
    def rfe_linear(X_train, Y_train, num_features, feature_names, step=30):
        rfe = RFE(estimator=RidgeClassifier(random_state=2, alpha=5), n_features_to_select=num_features, step=step)
        rfe.fit(X_train, Y_train)
        features = rfe.get_feature_names_out(feature_names)
        return features

    def parallel_filtering(self, X_train, Y_train, num_features, feature_names, split_ind, plot=True,
                           plot_num_features=20):
        
        results = {}
        filter_names = ("FRatio", "SymmetricUncertainty", "SpearmanCorr", "PearsonCorr", "Chi2", "Anova",
                        "LaplacianScore", "InformationGain", "KendallCorr", "FechnerCorr")
        filter_names = random.sample(filter_names, self.num_filters)
        multivariate = ("STIR", "TraceRatioFisher")
        filter_unsupervised = ("TraceRatioLaplacian",)

        arg_univariate = [(X_train, Y_train, num_features, feature_names, x) for x in filter_names]
        arg_multivariate = [(X_train, Y_train, num_features, feature_names, x) for x in multivariate]
        arg_unsupervised = [(X_train, num_features, feature_names, x) for x in filter_unsupervised]
        with get_context("spawn").Pool(self.num_thread) as pool:
            for num, res in enumerate(pool.starmap(self.univariate, arg_univariate)):
                print(f"univariate filter: {filter_names[num]}")
                results[filter_names[num]] = res

            for num, res in enumerate(pool.starmap(self.multivariate, arg_multivariate)):
                print(f"multivariate filter: {multivariate[num]}")
                results[multivariate[num]] = res

            for num, res in enumerate(pool.starmap(self.unsupervised, arg_unsupervised)):
                print(f"unsupervised filter: {filter_unsupervised[num]}")
                results[filter_unsupervised[num]] = res

        results["xgbtree"] = self.xgbtree(X_train, Y_train, feature_names, split_ind, plot, plot_num_features)
        results["random"] = self.random_classification(X_train, Y_train, feature_names)

        results = pd.concat(results)
        return results
    
    def _get_num_feature_range(self, num_features_min=None, num_features_max=None, step_range=None):
            if not num_features_min:
                num_features_min = int(len(self.features.index) / 10)
                if not num_features_max:
                    num_features_max = int(len(self.features.index) / 2)
                if not step_range:
                    step_range = int((num_features_max - num_features_min) / 5)
                num_feature_range = range(num_features_min, num_features_max, step_range)
            elif num_features_min and step_range and num_features_max:
                num_feature_range = range(num_features_min, num_features_max, step_range)
            else:
                num_feature_range = [num_features_min]
        
            return num_feature_range

    def _construct_features(self, features, feature_dict, num_feature_range, X_train, Y_train, num_features_max, i, 
                           plot=True, plot_num_features=20, rfe_step=30):
            transformed, scaler_dict = scale(self.scaler, X_train)
            # for each split I do again feature selection and save all the features selected from different splits
            # but with the same selector in the same dictionary
            print("filtering the features")
            ordered_features = self.parallel_filtering(transformed, Y_train, num_features_max, X_train.columns, i,
                                                       plot, plot_num_features)
            for num_features in num_feature_range:
                print(f"generating a feature set of {num_features} dimensions")
                for filters in ordered_features.index.get_level_values(0).unique():
                    feat = ordered_features.loc[filters]
                    feature_dict[f"{filters}_{num_features}"][f"split_{i}"] = features[feat.index[:num_features]]
                rfe_results = self.rfe_linear(transformed, Y_train, num_features, X_train.columns, rfe_step)
                feature_dict[f"rfe_{num_features}"][f"split_{i}"] = features[rfe_results]
            

    def construct_feature_set_kfold(self, num_features_min=None, num_features_max=None, step_range=None, rfe_step=30,
                              plot=True, plot_num_features=20):
        random.seed(self.seed)
        features = self.preprocess()
        skf = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.seed)
        feature_dict = defaultdict(dict)
        num_feature_range = self._get_num_feature_range(num_features_min, num_features_max, step_range)
        
        for i, (train_index, test_index) in enumerate(skf.split(self.features, self.label)):
            print(f"kfold {i}")
            X_train = features.iloc[train_index]
            Y_train = self.label.iloc[train_index].values.ravel()
            self._construct_features(features, feature_dict, num_feature_range, X_train, Y_train, 
                                    num_features_max, i, plot, plot_num_features, rfe_step)

        final_dict = {key: pd.concat(value, axis=1) for key, value in feature_dict.items()}
        with pd.ExcelWriter(self.excel_file, mode="w", engine="openpyxl") as writer:
            for key in final_dict.keys():
                write_excel(writer, final_dict[key], key)

    def construct_feature_set_holdout(self, num_features_min=None, num_features_max=None, step_range=None,
                                      plot=True, plot_num_features=20, rfe_step=30):
        random.seed(self.seed)
        features = self.preprocess()
        feature_dict = defaultdict(dict)
        num_feature_range = self._get_num_feature_range(num_features_min, num_features_max, step_range)

        X_train, X_test, Y_train, Y_test = train_test_split(features, self.label, test_size=0.20, random_state=self.seed, stratify=self.label)
        self._construct_features(features, feature_dict, num_feature_range, X_train, Y_train, num_features_max, 0, 
                                plot, plot_num_features, rfe_step)
        
        # TODO: Maybe change it to list(valu.values())[0] so It is not a multiindex column
        final_dict = {key: pd.concat(value, axis=1) for key, value in feature_dict.items()}
        with pd.ExcelWriter(self.excel_file, mode="w", engine="openpyxl") as writer:
            for key in final_dict.keys():
                write_excel(writer, final_dict[key], key)


def main():
    features, label, variance_threshold, feature_range, num_thread, scaler, excel_file, kfold, rfe_steps, plot, \
        plot_num_features, num_filters, seed, strategy = arg_parse()
    num_split, test_size = int(kfold.split(":")[0]), float(kfold.split(":")[1])
    feature_range = feature_range.split(":")
    num_features_min, num_features_max, step = feature_range
    if num_features_min == "none":
        num_features_min = None
        step = None
        num_features_max = None
    else:
        num_features_min = int(num_features_min)
        if step.isdigt():
            step = int(step)
        else:
            step = None
        if num_features_max.isdigit():
            num_features_min = int(num_features_max)
        else:
            num_features_max = None
    selection = FeatureSelection(label, excel_file, features, variance_threshold, num_thread, scaler,
                                 num_split, test_size, num_filters, seed)
    if strategy == "holdout":
        selection.construct_feature_set_holdout(num_features_min, num_features_max, step, rfe_steps, plot, plot_num_features)
    elif strategy == "kfold":
        selection.construct_feature_set_kfold(num_features_min, num_features_max, step, rfe_steps, plot, plot_num_features)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()