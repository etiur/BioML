from dataclasses import dataclass, field
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from functools import cached_property
import argparse
from scipy.spatial import distance
from Bio import SeqIO
import Bio
from Bio.SeqIO import FastaIO
import numpy as np
import pandas as pd
from pathlib import Path
from ..utilities.utils import scale, read_outlier_file
from .base import DataParser


def arg_parse():
    parser = argparse.ArgumentParser(description="Predict using the models and average the votations")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path", required=True)
    parser.add_argument("-tr", "--training_features", required=True,
                        help="The file to where the training features are saved in excel or csv format")
    parser.add_argument("-sc", "--scaler", default="zscore", choices=("robust", "zscore", "minmax"),
                        help="Choose one of the scaler available in scikit-learn, defaults to zscore")
    parser.add_argument("-m", "--model_path", required=True,
                        help="The directory for the generated models")
    parser.add_argument("-te", "--test_features", required=True,
                        help="The file to where the test features are saved in excel or csv format")
    parser.add_argument("-d", "--res_dir", required=False,
                        help="The file where the extracted features from the new data are stored",
                        default="prediction_results")
    parser.add_argument("-nss", "--number_similar_samples", required=False, default=1, type=int,
                        help="The number of similar training samples to filter the predictions")
    parser.add_argument("-otr", "--outliers_train", nargs="+", required=False, default=(),
                        help="A list of outliers if any, the name should be the same as the index of "
                             " training features, you can also specify the path to a file in plain text format, each "
                             "record should be in a new line")
    parser.add_argument("-ote", "--outliers_test", nargs="+", required=False, default=(),
                        help="A list of outliers if any, the name should be the same as the index of "
                             " test features, you can also specify the path to a file in plain text format, each "
                             "record should be in a new line")
    parser.add_argument("-p", "--problem", required=False, 
                        default="classification", choices=("classification", "regression"), help="The problem type")
    parser.add_argument("-l", "--label", required=False, default=None,
                        help="Use it if the lables is in the training features so it is removed, but not necessary otherwise")
    parser.add_argument("-ad", "--applicability_domain", required=False, action="store_false", 
                        help="If to use the applicability domain to filter the predictions")
    parser.add_argument("-sh", "--sheet_name", required=False, default=None, 
                        help="The sheet name for the excel file if the training features is in excel format")
    args = parser.parse_args()

    return [args.fasta_file, args.training_features, args.scaler, args.model_path, args.test_features,
             args.res_dir, args.number_similar_samples, args.outlier_train, args.outlier_test, args.problem, args.label,
             args.applicability_domain, args.sheet_name]

@dataclass
class Predictor:
    """
    A class to perform predictions
    Initialize the class with the test features, the experiment and the model path

    Parameters
    ____________
    test_features: pd.DataFrame
        The test features
    model: RegressionExperiment | ClassificationExperiment
        The experiment object
    model_path: str | Path
        The path to the model
    """

    test_features: pd.DataFrame
    model: RegressionExperiment | ClassificationExperiment
    model_path: str | Path

    @cached_property
    def loaded_model(self):
        return self.model.load_model(self.model_path, verbose=False)
        

    def predicting(self, problem: str="classification") -> pd.DataFrame:
        """
        Make predictions on new samples

        Parameters
        ___________
        problem: str
            The problem type, either classification or regression

        Returns
        -------
        pd.DataFrame
            The predictions appended as new columns to the test set
        """
        if problem == "classification":
            return self.model.predict_model(self.loaded_model, self.test_features, verbose=False, raw_score=True)
        return self.model.predict_model(self.loaded_model, self.test_features, verbose=False)


@dataclass(slots=True)
class ApplicabilityDomain:
    """
    A class that looks for the applicability domain
    """
    x_train: pd.DataFrame = field(default=None, init=False, repr=False)
    x_test: pd.DataFrame = field(default=None, init=False, repr=False)
    thresholds: list = field(default=None, init=False, repr=False) #
    n_insiders: list = field(default_factory=list, init=False, repr=False)

    def fit(self, x_train: pd.DataFrame) -> np.ndarray:
        """
        A function to calculate the training sample threshold for the applicability domain.

        Parameters
        ----------
        x_train : pandas DataFrame object
            The training data.

        Returns
        -------
        np.ndarray
            An array of the computed thresholds.
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
        all_d = [sum(all_allowed[i]) for i, _ in enumerate(d_no_ii)]
        self.thresholds = np.divide(all_d, n_allowed)  # threshold computation
        self.thresholds[np.isinf(self.thresholds)] = min(self.thresholds)  # setting to the minimum value where infinity
        return self.thresholds

    def predict(self, x_test: pd.DataFrame) -> list:
        """
        A function to find those samples that are within the training samples' threshold

        Parameters
        ___________
        x_test: pandas Dataframe object

        Returns
        __________
        list
            The number of training samples within the applicability domain for each test sample
        """
        # calculating the distance of test with each of the training samples
        d_train_test = np.array([distance.cdist(np.array(x).reshape(1, -1), self.x_train) for x in x_test])
        for i in d_train_test:  # for each sample
            # saving indexes of training with distance < threshold
            idxs = [j for j, d in enumerate(i[0]) if d <= self.thresholds[j]]
            self.n_insiders.append(len(idxs))  # for each test sample see how many training samples is within the AD

        return self.n_insiders

    def filter_model_predictions(self, predictions: pd.DataFrame, min_num: int=1) -> pd.DataFrame:
        """
        Eliminate the models individual predictions of sequences that did not pass the applicability domain threshold

        Parameters
        ----------
        predictions: pd.Dataframe
            Prediction from all the classifiers
        path_name: str | Path
            The path where to save the filtered predictions
        min_num: int
            The minimum number to be considered of the same applicability domain

        Returns
        --------
        object: pd.Dataframe
            The predictions of each of the models kept in the dataframe, the columns are the different model predictions
            and the rows the different sequences (The feature columns are removed from the predictions)
        """
  
        filtered_index = [f"sample_{x}" for x, similarity_score in enumerate(self.n_insiders) if similarity_score >= min_num]
        filtered_names = [pred_index for _, (pred_index, similarity_score) in enumerate(zip(predictions.index, self.n_insiders)) if similarity_score >= min_num]
        filtered_pred = predictions.loc[filtered_names]
        filtered_n_insiders = pd.Series([d for _, d in enumerate(self.n_insiders) if d >= min_num], name="AD_number", index=filtered_names)
        pred = pd.concat([filtered_pred, filtered_n_insiders], axis=1)
        pred.index = filtered_index
 
        return pred

@dataclass(slots=True)
class FastaExtractor:
    """
    A class that extracts sequences from a FASTA file according to the predictions and 
    saves the results in a specified directory.

    Attributes
    ----------
    fasta_file : str or Path
        The path to the input FASTA file.
    resdir : str or Path, optional
        The path to the directory where the extracted sequences will be saved. Default is "prediction_results".
    """
    fasta_file: str | Path
    resdir: str | Path = "prediction_results"

    def __post_init__(self):
        self.fasta_file = Path(self.fasta_file)
        self.resdir = Path(self.resdir)

    def separate_negative_positive(self, pred: pd.DataFrame):
        """
        Parameters
        ______________
        pred: pd.DataFrame
            The predictions, the index should be sample_index, where index starts from 0 to len(pred)

        Return
        ________
        positive: list[SeqIO.SeqRecord]
        negative: list[SeqIO.SeqRecord]
        """
        # separating the records according to if the prediction is positive or negative
        
        if (self.fasta_file.parent / "no_short.fasta").exists():
            self.fasta_file = self.fasta_file.parent / "no_short.fasta"

        with open(self.fasta_file) as inp:
            record = SeqIO.parse(inp, "fasta")
            p = 0
            positive = []
            negative = []
            for ind, seq in enumerate(record):
                try:
                    if int(pred.index[p].split("_")[1]) == ind:
                        col = pred.iloc[p]
                        seq.id = f"{seq.id}-###label:{col['prediction_label']}"
                        if col.index.str.contains("prediction_score").any():
                            seq.id = f"{seq.id}-###prob_0:{col['prediction_score_0']}-###prob_1:{col['prediction_score_1']}"
                        if col.index.str.contains("AD_number").any():
                            seq.id = f"{seq.id}-###AD:{col['AD_number']}"
                        if pred["prediction_label"].iloc[p] == 1:
                            positive.append(seq)
                        else:
                            negative.append(seq)
                        p += 1
                except IndexError:
                    break

        return positive, negative
    
    @staticmethod    
    def _sorting_function(sequence: Bio.SeqRecord.SeqRecord):
        scores = []
        id_ = sequence.id.split("-###")
        for x in id_:
            if "prob_1" in x:
                scores.append(float(x.split(":")[1]))
            if "AD" in x:
                scores.append(int(float((x.split(":")[1]))))
            if "label" in x:
                scores.append(float(x.split(":")[1]))
        return tuple(scores)

    def extract(self, positive_list: list[Bio.SeqRecord.SeqRecord], negative_list: list[Bio.SeqRecord.SeqRecord], 
                positive_fasta: str="positive.fasta", negative_fasta: str="negative.fasta"):
        """
        A function to extract those test fasta sequences that passed the filter

        Parameters
        ___________
        positive_list: list[Bio.SeqRecord.SeqRecord]
            The positive class sequences
        negative_list: list[Bio.SeqRecord.SeqRecord]
            The negative class sequences
        positive_fasta: str, optional
            The new filtered fasta file with positive predictions
        negative_fasta: str, optional
            The new filtered fasta file with negative sequences
        """
        # writing the positive and negative fasta sequences to different files
        self.resdir.mkdir(exist_ok=True, parents=True)
        with open(f"{self.resdir}/{positive_fasta}", "w") as pos:
            positive = sorted(positive_list, reverse=True, key=self._sorting_function)
            fasta_pos = FastaIO.FastaWriter(pos, wrap=None)
            fasta_pos.write_file(positive)
        with open(f"{self.resdir}/{negative_fasta}", "w") as neg:
            negative = sorted(negative_list, reverse=True, key=self._sorting_function)
            fasta_neg = FastaIO.FastaWriter(neg, wrap=None)
            fasta_neg.write_file(negative)


def predict(test_features: pd.DataFrame, model_path: str | Path, problem: str="classification") -> pd.DataFrame:
    """
    Make predictions on new samples.

    Parameters
    ----------
    test_features : pandas DataFrame object
        The test data.
    model_path : str or Path
        The path to the trained model.
    problem : str, default="classification"
        The type of problem. Must be one of "classification" or "regression".

    Returns
    -------
    pd.DataFrame
        The predicted values appended to the test features.

    Notes
    -----
    This function makes predictions on new samples. 
    It takes in a pandas DataFrame object `test_features` as the test data, a string `model_path` as the path to the trained model,
    and a string `problem` as the type of problem. 
    The function creates an instance of either the `ClassificationExperiment` or `RegressionExperiment` 
    class based on the `problem` parameter. It then creates an instance of the `Predictor` class with the test data, experiment, 
    and model path as parameters. The function calls the `predicting` method of the `Predictor` object to make predictions on the 
    test data. The function returns the predicted values as a numpy array.

    Examples
    --------
    >>> from predict import predict
    >>> import pandas as pd
    >>> import numpy as np
    >>> test_features = pd.DataFrame(np.random.rand(10, 5))
    >>> model_path = 'model.pkl'
    >>> pred = predict(test_features, model_path, problem='classification')
    """
    if problem == "classification":
        experiment = ClassificationExperiment()
    elif problem == "regression":
        experiment = RegressionExperiment()

    predictor = Predictor(test_features, experiment, model_path)
    pred = predictor.predicting(problem)
    return pred


def domain_filter(predictions: pd.DataFrame, scaled_training_features: pd.DataFrame, scaled_test_features: pd.DataFrame,
                  min_num: int=1) -> pd.DataFrame:
    """
    Filter predictions using the applicability domain.

    Parameters
    ----------
    predictions : pd.DataFrame object
        The predicted values.
    scaled_training_features : pandas DataFrame object
        The scaled training data.
    scaled_test_features : pandas DataFrame object
        The scaled test data.
    min_num : int, default=1
        The minimum number of samples required to be within the applicability domain.

    Returns
    -------
    pd.DataFrame
        The filtered predictions.

    Notes
    -----
    This function filters predictions using the applicability domain. 
    It takes in the predicted values as a numpy ndarray, the scaled training data as a pandas DataFrame object, 
    the scaled test data as a pandas DataFrame object, the path to the directory to save the filtered predictions as a string,
      and the minimum number of samples required to be within the applicability domain as an integer. 
      The function creates an instance of the `ApplicabilityDomain` class and fits it to the scaled training data. 
      It then predicts the applicability domain for the scaled test data. The function calls the `filter_model_predictions` 
      method of the `ApplicabilityDomain` object to filter the predictions based on the applicability domain. 
      The function returns the filtered predictions as a pd.DataFrame object.

    Examples
    --------
    >>> from predict import domain_filter
    >>> import pandas as pd
    >>> import numpy as np
    >>> predictions = np.random.rand(10)
    >>> scaled_training_features = pd.DataFrame(np.random.rand(10, 5))
    >>> scaled_test_features = pd.DataFrame(np.random.rand(5, 5))
    >>> filtered_predictions = domain_filter(predictions, scaled_training_features, 
    >>> scaled_test_features, res_dir="results", min_num=2)
    """
    domain = ApplicabilityDomain()
    domain.fit(scaled_training_features)
    domain.predict(scaled_test_features)
    # return the prediction after the applicability domain filter of SVC (the filter depends on the feature space)
    pred = domain.filter_model_predictions(predictions, min_num)

    return pred


def main():
    fasta, training_features, scaler, model_path, test_features, res_dir, number_similar_samples, \
    outlier_train, outlier_test, problem, label, applicability_domain, sheet_name = arg_parse()

    # read outliers
    outlier_test = read_outlier_file(outlier_test)
    outlier_train = read_outlier_file(outlier_train)

    # preparing the prediction
    feature = DataParser(training_features, label, outliers=outlier_train, sheets=sheet_name)
    test_features = feature.remove_outliers(feature.read_features(test_features), outlier_test)
    predictions = predict(test_features, model_path, problem)
    if applicability_domain:
        transformed, _, test_x = scale(scaler, feature.drop(), test_features)
        predictions = domain_filter(predictions, transformed , test_x, number_similar_samples)
    else:
        test_index = [f"sample_{x}" for x, _ in enumerate(predictions.index)]
        predictions.index = test_index
    
    # save the predictions
    col_name = ["prediction_score", "prediction_label", "AD_number"]
    predictions = predictions.loc[:, predictions.columns.str.contains("|".join(col_name))]
    Path(res_dir).mkdir(exist_ok=True, parents=True)
    predictions.to_csv(f"{res_dir}/predictions.csv")

    if problem == "classification":
        extractor = FastaExtractor(fasta, res_dir)
        positive, negative = extractor.separate_negative_positive(predictions)
        extractor.extract(positive, negative, positive_fasta="positive.fasta", negative_fasta="negative.fasta")   

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
