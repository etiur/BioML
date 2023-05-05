import joblib
import argparse
from scipy.spatial import distance
from Bio import SeqIO
from Bio.SeqIO import FastaIO
from utilities import scale
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def arg_parse():
    parser = argparse.ArgumentParser(description="Generate the models from the ensemble")

    parser.add_argument("-e", "--excel", required=False,
                        help="The file to where the selected features are saved in excel format",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-sc", "--scaler", default="robust", choices=("robust", "standard", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to RobustScaler")
    parser.add_argument("-o", "--model_output", required=False,
                        help="The directory for the generated models",
                        default="models")
    parser.add_argument("-va", "--prediction_threshold", required=False, default=1.0, type=float,
                        help="Between 0.5 and 1 and determines what considers to be a positive prediction, if 1 only"
                             "those predictions where all models agrees are considered to be positive")
    parser.add_argument("-s", "--sheets", required=True, nargs="+",
                        help="Names or index of the selected sheets for both features and hyperparameters and the "
                             "index of the models in this format-> sheet (name, index):index model1,index model2 "
                             "without the spaces. If only index or name of the sheets, it is assumed that all kfold models "
                             "are selected. It is possible to have one sheet with kfold indices but in another ones "
                             "without")
    parser.add_argument("-ne", "--extracted", required=False,
                        help="The file where the extracted features from the new data are stored",
                        default="extracted_features/new_features.xlsx")
    args = parser.parse_args()

    return [args.excel, args.scaler, args.model_output, args.prediction_threshold, args.sheets, args.extracted]



class EnsembleVoting:
    """
    A class to perform ensemble voting
    """

    def __init__(self, sheets, extracted_out="extracted_features/new_features.xlsx", models="models",
                 selected_features="training_features/selected_features.xlsx", scaler="robust", prediction_threshold=1):
        """
        Initialize the class EnsembleVoting

        Parameters
        ____________
        extracted_out: str
            The path to the directory where the new extracted feature files are
        """
        self.extracted_out = Path(extracted_out)
        self.training_features = Path(selected_features)
        self.models = Path(models)
        self.scaler = scaler
        self.prediction_threshold = prediction_threshold

        if sheets:
            self.selected_sheets = []
            self.selected_kfolds = {}
            self._check_sheets(sheets)

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

    def _check_features(self):
        with_split = True
        features = pd.read_excel(self.training_features, index_col=0, sheet_name=self.selected_sheets, header=[0, 1],
                                 engine='openpyxl')
        new_features = pd.read_excel(self.extracted_out, index_col=0, sheet_name=self.selected_sheets, header=[0, 1],
                                     engine='openpyxl')
        if f"split_{0}" not in list(features.values())[0].columns.unique(0):
            with_split = False
            features = pd.read_excel(self.training_features, index_col=0, sheet_name=self.selected_sheets, header=0,
                                     engine='openpyxl')
            new_features = pd.read_excel(self.extracted_out, index_col=0, sheet_name=self.selected_sheets, header=0,
                                         engine='openpyxl')

        return features, with_split, new_features

    def get_models(self):
        model_dict = defaultdict(dict)
        models = self.models.glob("*/*.joblib")
        for mod in models:
            sheet = mod.parent.name
            split_ind = int(mod.stem.split("_")[-1])
            model_dict[sheet][split_ind] = joblib.load(mod)
        return model_dict

    def predicting(self):
        """
        Make predictions on new samples
        """
        prediction = defaultdict(dict)
        models = self.get_models()
        # extract the features
        features, with_split, new_features = self._check_features()
        for sheet, split_dict in models.items():
            for ind, mod in split_dict.items():
                if with_split:
                    sub_feat = features[sheet].loc[:, f"split_{ind}"]
                    sub_new_feat = new_features[sheet].loc[:, f"split_{ind}"]
                else:
                    sub_feat = features[sheet]
                    sub_new_feat = new_features[sheet]
                old_feat_scaled, scaler_dict, new_feat_scaled = scale(self.scaler, sub_feat, sub_new_feat)
                pred = mod.predict(new_feat_scaled)
                name = mod.__class__.__name__
                prediction[sheet][f"{name}_{ind}"] = pred

        return prediction

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


class ApplicabilityDomain():
    """
    A class that looks for the applicability domain
    """

    def __init__(self, sheets, selected_features="training_features/selected_features.xlsx"):
        """
        Initialize the class
        """
        self.x_train = None
        self.x_test = None
        self.thresholds = None
        self.test_names = None
        self.pred = []
        self.dataframe = None
        self.n_insiders = []
        path_to_esterase = "/gpfs/projects/bsc72/ruite/enzyminer/data/final_features.xlsx"
        x_svc = pd.read_excel(f"{path_to_esterase}", index_col=0, sheet_name=f"ch2_30", engine='openpyxl')
        self.ad_indices = []
        self.selected_sheets = []
        self.selected_kfolds = {}
        self._check_sheets(sheets)
        self.feature_path = Path(selected_features)
        self.training_names = x_svc.index

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
    def _check_features(self):
        with_split = True
        features = pd.read_excel(self.feature_path, index_col=0, sheet_name=self.selected_sheets, header=[0, 1],
                                 engine='openpyxl')
        if f"split_{0}" not in list(features.values())[0].columns.unique(0):
            with_split = False
            features = pd.read_excel(self.feature_path, index_col=0, sheet_name=self.selected_sheets, header=0,
                                     engine='openpyxl')
        return features, with_split
    def fit(self, x_train):
        """
        A function to calculate the training sample threshold for the applicability domain

        Parameters
        ___________
        x_train: pandas Dataframe object
        """
        self.x_train = x_train
        # for each of the training sample calculate the distance to the other samples
        distances = np.array([distance.cdist(np.array(x).reshape(1, -1), self.x_train) for x in self.x_train])
        distances_sorted = [np.sort(d[0]) for d in distances]
        d_no_ii = [d[1:] for d in distances_sorted]  # not including the distance with itself, which is 0
        k = int(round(pow(len(self.x_train), 1 / 3)))
        d_means = [np.mean(d[:k]) for d in d_no_ii]  # mean values, np.mean(d[:k][0])
        Q1 = np.quantile(d_means, .25)
        Q3 = np.quantile(d_means, .75)
        d_ref = Q3 + 1.5 * (Q3 - Q1)  # setting the reference value
        n_allowed = []
        all_allowed = []
        for i in d_no_ii:
            d_allowed = [d for d in i if d <= d_ref]  # keeping the distances that are smaller than the ref value
            all_allowed.append(d_allowed)
            n_allowed.append(len(d_allowed))  # calculating the density (number of distances kept) per sample

        # selecting minimum value not 0:
        min_val = [np.sort(n_allowed)[i] for i in range(len(n_allowed)) if np.sort(n_allowed)[i] != 0]

        # replacing 0's with the min val
        n_allowed = [n if n != 0 else min_val[0] for n in n_allowed]
        all_d = [sum(all_allowed[i]) for i, d in enumerate(d_no_ii)]
        self.thresholds = np.divide(all_d, n_allowed)  # threshold computation
        self.thresholds[np.isinf(self.thresholds)] = min(self.thresholds)  # setting to the minimum value where infinity
        return self.thresholds

    def predict(self, x_test):
        """
        A function to find those samples that are within the training samples' threshold

        Parameters
        ___________
        x_test: pandas Dataframe object
        """
        self.x_test = x_test
        self.test_names = ["sample_{}".format(i) for i in range(self.x_test.shape[0])]
        # calculating the distance of test with each of the training samples
        d_train_test = np.array([distance.cdist(np.array(x).reshape(1, -1), self.x_train) for x in self.x_test])
        for i in d_train_test:  # for each sample
            # saving indexes of training with distance < threshold
            idxs = [self.training_names[j] for j, d in enumerate(i[0]) if d <= self.thresholds[j]]
            self.n_insiders.append(len(idxs))  # for each test sample see how many training samples is within the AD
            idxs = "_".join(idxs)
            self.ad_indices.append(idxs)

        return self.n_insiders, self.ad_indices

    def _filter_models_vote(self, svc, knn, ridge, filtered_names, min_num=1):
        """
        Eliminate the models individual predictions of sequences that did not pass the applicability domain threshold

        Parameters
        ----------
        svc: dict
            Predictions from SVCs
        knn: dict
            Predictions from KNNs
        ridge: dict
            Prediction from ridge classifiers
        filtered_names: list[str]
            Names of the test samples after the filtering
        min_num: int
            The minimum number to be considered of the same applicability domain

        Returns
        --------
        object: pd.Dataframe
            The predictions of each of the models kept in the dataframe, the columns are the different model predictions
            and the rows the different sequences
        """
        results = {}
        for s, pred in svc.items():
            sv = [d[0] for x, d in enumerate(zip(pred, self.n_insiders)) if d[1] >= min_num]
            results[s] = sv
        for s, pred in ridge.items():
            sv = [d[0] for x, d in enumerate(zip(pred, self.n_insiders)) if d[1] >= min_num]
            results[s] = sv
        for s, pred in knn.items():
            sv = [d[0] for x, d in enumerate(zip(pred, self.n_insiders)) if d[1] >= min_num]
            results[s] = sv

        return pd.DataFrame(results, index=filtered_names)

    def filter(self, prediction, svc, knn, ridge, min_num=1, path_name="filtered_predictions.parquet"):
        """
        Filter those predictions that have less than min_num training samples

        Parameters
        ___________
        prediction: array
            An array of the average of predictions
        svc: dict[array]
            The prediction of the SVC models
        knn: dict[array]
            The predictions of the different Knn models
        ridge: dict[array]
            The predictions of the different ridge models
        index: array
            The index of those predictions that were not unanimous between different models
        path_name: str, optional
            The path for the csv file
        min_num: int, optional
            The minimun number of training samples within the AD of the test samples
        """
        # filter the predictions and names based on the if it passed the threshold of similar samples
        filtered_pred = [d[0] for x, d in enumerate(zip(prediction, self.n_insiders)) if d[1] >= min_num]
        filtered_names = [d[0] for y, d in enumerate(zip(self.test_names, self.n_insiders)) if d[1] >= min_num]
        filtered_n_insiders = [d for s, d in enumerate(self.n_insiders) if d >= min_num]
        # Turn the different arrays into pandas Series or dataframes
        pred = pd.Series(filtered_pred, index=filtered_names)
        n_applicability = pd.Series(filtered_n_insiders, index=filtered_names)
        models = self._filter_models_vote(svc, knn, ridge, filtered_names, min_num)
        # concatenate all the objects
        self.pred = pd.concat([pred, n_applicability, models], axis=1)
        self.pred.columns = ["prediction", "AD_number"] + list(models.columns)
        self.pred.to_parquet(path_name)
        return self.pred

    def separate_negative_positive(self, fasta_file, pred=None):
        """
        Parameters
        ______________
        fasta_file: str
            The input fasta file
        pred: list, optional
            The predictions

        Return
        ________
        positive: list[Bio.SeqIO]
        negative: list[Bio.SeqIO]
        """
        if pred is not None:
            self.pred = pred
        # separating the records according to if the prediction is positive or negative
        if dirname(fasta_file) != "":
            base = dirname(fasta_file)
        else:
            base = "."
        with open(f"{base}/no_short.fasta") as inp:
            record = SeqIO.parse(inp, "fasta")
            p = 0
            positive = []
            negative = []
            for ind, seq in enumerate(record):
                try:
                    if int(self.pred.index[p].split("_")[1]) == ind:
                        col = self.pred[self.pred.columns[2:]].iloc[p]
                        mean = round(sum(col) / len(col), 2)
                        col = [f"{col.index[i]}-{d}" for i, d in enumerate(col)]
                        seq.id = f"{seq.id}-{'+'.join(col)}-###prob:{mean}###AD:{self.pred['AD_number'][p]}"
                        if self.pred["prediction"][p] == 1:
                            positive.append(seq)
                        else:
                            negative.append(seq)
                        p += 1
                except IndexError:
                    break

        return positive, negative

    def find_max_ad(self, pred1, pred2, pred3):
        """
        find the maximum applicability domain of the 2 preds
        parameters
        ___________
        pred1: array
            svc predictions
        pred2: array
            knn predictions
        pred3: array
            ridge predictions
        """
        assert len(pred1) == len(pred2) == len(pred3), "The predictions have different lengths"
        ad = []
        pred = pred1.copy()
        for idx in pred1.index:
            if pred3["AD_number"].loc[idx] <= pred1["AD_number"].loc[idx] >= pred2["AD_number"].loc[idx]:
                ad.append(f'{pred1["AD_number"].loc[idx]}-svc')
            elif pred1["AD_number"].loc[idx] <= pred2["AD_number"].loc[idx] >= pred3["AD_number"].loc[idx]:
                ad.append(f'{pred2["AD_number"].loc[idx]}-knn')
            else:
                ad.append(f'{pred3["AD_number"].loc[idx]}-ridge')
        pred["AD_number"] = ad
        return pred

    def extract(self, fasta_file, pred1=None, pred2=None, pred3=None, positive_fasta="positive.fasta",
                negative_fasta="negative.fasta", res_dir="results"):
        """
        A function to extract those test fasta sequences that passed the filter

        Parameters
        ___________
        fasta_file: str
            The path to the test fasta sequences
        pred: pandas Dataframe, optional
            Predictions
        positive_fasta: str, optional
            The new filtered fasta file with positive predictions
        negative_fasta: str, optional
            The new filtered fasta file with negative sequences
        res_dir: str, optional
            The folder where to keep the prediction results
        """
        if pred2 is not None and pred3 is not None:
            pred1 = self.find_max_ad(pred1, pred2, pred3)
        positive, negative = self.separate_negative_positive(fasta_file, pred1)
        # writing the positive and negative fasta sequences to different files
        with open(f"{res_dir}/{positive_fasta}", "w") as pos:
            positive = sorted(positive, reverse=True, key=lambda x: (float(x.id.split("###")[1].split(":")[1]),
                                                                     int(x.id.split("###")[2].split(":")[1].split("-")[0])))
            fasta_pos = FastaIO.FastaWriter(pos, wrap=None)
            fasta_pos.write_file(positive)
        with open(f"{res_dir}/{negative_fasta}", "w") as neg:
            negative = sorted(negative, reverse=True, key=lambda x: (float(x.id.split("###")[1].split(":")[1]),
                                                                     int(x.id.split("###")[2].split(":")[1].split("-")[0])))
            fasta_neg = FastaIO.FastaWriter(neg, wrap=None)
            fasta_neg.write_file(negative)