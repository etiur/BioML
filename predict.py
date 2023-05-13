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
    parser = argparse.ArgumentParser(description="Predict using the models and average the votations")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path", required=True)
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
    parser.add_argument("-ne", "--extracted", required=False,
                        help="The file where the extracted features from the new data are stored",
                        default="extracted_features/new_features.xlsx")
    parser.add_argument("-d", "--res_dir", required=False,
                        help="The file where the extracted features from the new data are stored",
                        default="prediction_results")
    parser.add_argument("-nss", "--number_similar_samples", required=False, default=1, type=int,
                        help="The number of similar training samples to filter the predictions")
    args = parser.parse_args()

    return [args.fasta, args.excel, args.scaler, args.model_output, args.prediction_threshold, args.extracted,
            args.res_dir, args.number_similar_samples]


class EnsembleVoting:
    """
    A class to perform ensemble voting
    """

    def __init__(self, extracted_features="extracted_features/new_features.xlsx", models="models",
                 selected_features="training_features/selected_features.xlsx", scaler="robust"):
        """
        Initialize the class EnsembleVoting

        Parameters
        ____________
        extracted_out: str
            The path to the directory where the new extracted feature files are
        """
        self.extracted_out = Path(extracted_features)
        self.training_features = Path(selected_features)
        self.models_path = Path(models)
        self.scaler = scaler
        self.with_split = True
        self.models_dict = self.get_models()
        self.selected_sheets = list(self.models_dict.keys())

    def _check_features(self):
        features = pd.read_excel(self.training_features, index_col=0, header=[0, 1], engine='openpyxl')
        if f"split_{0}" not in features.columns.unique(0):
            self.with_split = False

        if self.with_split:
            features = pd.read_excel(self.training_features, index_col=0, sheet_name=self.selected_sheets,
                                     header=[0, 1], engine='openpyxl')
            new_features = pd.read_excel(self.extracted_out, index_col=0, sheet_name=self.selected_sheets,
                                         header=[0, 1], engine='openpyxl')
        else:
            features = pd.read_excel(self.training_features, index_col=0, sheet_name=self.selected_sheets, header=0,
                                     engine='openpyxl')
            new_features = pd.read_excel(self.extracted_out, index_col=0, sheet_name=self.selected_sheets, header=0,
                                         engine='openpyxl')

        return features, new_features

    def get_models(self):
        model_dict = defaultdict(dict)
        models = self.models_path.glob("*/*.joblib")
        for mod in models:
            sheet = mod.parent.name[6:]
            if sheet.isdigit():
                sheet = int(sheet)
            split_ind = int(mod.stem.split("_")[-1])
            model_dict[sheet][split_ind] = joblib.load(mod)
        return model_dict

    def predicting(self):
        """
        Make predictions on new samples
        """
        feature_dict = {}
        predictions = {}
        # extract the features
        features, new_features = self._check_features()
        for sheet, split_dict in self.models_dict.items():
            for ind, mod in split_dict.items():
                if self.with_split:
                    sub_feat = features[sheet].loc[:, f"split_{ind}"]
                    sub_new_feat = new_features[sheet].loc[:, f"split_{ind}"]
                else:
                    sub_feat = features[sheet]
                    sub_new_feat = new_features[sheet]
                old_feat_scaled, scaler_dict, new_feat_scaled = scale(self.scaler, sub_feat, sub_new_feat)
                pred = mod.predict(new_feat_scaled)
                name = mod.__class__.__name__
                predictions[f"{sheet}_{name}_{ind}"] = pred
                feature_dict[f"{sheet}_{name}_{ind}"] = (old_feat_scaled, new_feat_scaled)

        return predictions, feature_dict

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


class ApplicabilityDomain:
    """
    A class that looks for the applicability domain
    """

    def __init__(self):
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
        self.ad_indices = []

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
            idxs = [j for j, d in enumerate(i[0]) if d <= self.thresholds[j]]
            self.n_insiders.append(len(idxs))  # for each test sample see how many training samples is within the AD

        return self.n_insiders

    def _filter_models_vote(self, model_predictions, filtered_names, min_num=1):
        """
        Eliminate the models individual predictions of sequences that did not pass the applicability domain threshold

        Parameters
        ----------
        model_predictions: dict[dict[np.ndarray]]
            Prediction from all the classifiers
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
        for sheet, pred in model_predictions.items():
            filtered_pred = [d[0] for x, d in enumerate(zip(pred, self.n_insiders)) if d[1] >= min_num]
            results[sheet] = filtered_pred

        return pd.DataFrame(results, index=filtered_names)

    def filter(self, prediction, model_predictions, min_num=1, path_name: str | Path= "filtered_predictions.parquet"):
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
        path_name: Path | str, optional
            The path for the csv file
        min_num: int, optional
            The minimun number of training samples within the AD of the test samples
        """
        # filter the ensemble predictions and names based on if it passed the threshold of similar samples
        filtered_pred = [d[0] for x, d in enumerate(zip(prediction, self.n_insiders)) if d[1] >= min_num]
        filtered_names = [d[0] for y, d in enumerate(zip(self.test_names, self.n_insiders)) if d[1] >= min_num]
        filtered_n_insiders = [d for s, d in enumerate(self.n_insiders) if d >= min_num]
        # Turn the different arrays into pandas Series or dataframes
        pred = pd.Series(filtered_pred, index=filtered_names)
        n_applicability = pd.Series(filtered_n_insiders, index=filtered_names)
        models = self._filter_models_vote(model_predictions, filtered_names, min_num) # individual votes
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
        self.fasta_file = Path(fasta_file)
        if (self.fasta_file.parent / "no_short.fasta").exists():
            self.fasta_file = self.fasta_file.parent / "no_short.fasta"

        with open(self.fasta_file) as inp:
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

    def extract(self, fasta_file, pred=None, positive_fasta="positive.fasta", negative_fasta="negative.fasta",
                res_dir: str | Path = "results"):
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
        res_dir: Path | str, optional
            The folder where to keep the prediction results
        """
        positive, negative = self.separate_negative_positive(fasta_file, pred)
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


def vote_and_filter(fasta_file, extracted_features="extracted_features/new_features.xlsx", models="models",
                    selected_features="training_features/selected_features.xlsx", scaler="robust",
                    prediction_threshold=1, res_dir="prediction_results", min_num=1):

    res_dir = Path(res_dir)
    (res_dir / "domain").mkdir(parents=True, exist_ok=True)
    (res_dir / "positive").mkdir(parents=True, exist_ok=True)
    (res_dir / "negative").mkdir(parents=True, exist_ok=True)

    ensemble = EnsembleVoting(extracted_features, models, selected_features, scaler)
    # predictions
    predictions, feature_dict = ensemble.predicting()
    all_voting, all_index = ensemble.vote(prediction_threshold, *predictions.values())
    # applicability domain for each of the models
    domain_list = []
    for key, scaled_feature in feature_dict.items():
        domain = ApplicabilityDomain()
        domain.fit(scaled_feature[0])
        domain.predict(scaled_feature[1])
    # return the prediction after the applicability domain filter of SVC (the filter depends on the feature space)
        pred = domain.filter(all_voting, predictions, min_num, res_dir/"domain"/f"{key}.parquet")
        domain.extract(fasta_file, pred, positive_fasta=f"positive/{key}.fasta",
                       negative_fasta=f"negative/{key}.fasta", res_dir=res_dir)
        domain_list.append(pred)

    # Then filter again to see which sequences are within the AD of all the algorithms since it is an ensemble classifier
    name_set = set(domain_list[0].index).intersection(*tuple(x.index for x in domain_list[1:]))
    name_set = sorted(name_set, key=lambda x: int(x.split("_")[1]))
    common_domain = domain_list[0].loc[name_set]
    common_domain.to_parquet(f"{res_dir}/common_domain.parquet")
    # the positive sequences extracted will have the AD of the SVC
    domain.extract(fasta_file, common_domain, positive_fasta=f"common_positive.fasta",
                       negative_fasta=f"common_negative.fasta", res_dir=res_dir)

    def main():
        fasta, selected_features, scaler, model_output, prediction_threshold, extracted, res_dir, \
            number_similar_samples = arg_parse()
        vote_and_filter(fasta, extracted, model_output, selected_features, scaler, prediction_threshold,
                        res_dir, number_similar_samples)

    if __name__ == "__main__":
        # Run this if this file is executed from command line but not if is imported as API
        main()
