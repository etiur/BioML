import pandas as pd
from .base import PycaretInterface, Trainer, DataParser
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class Classifier(Trainer):
    def __init__(self, model: PycaretInterface, training_output="training_results", num_splits=5, test_size=0.2,
                 outliers: tuple[str, ...]=(), scaler="robust",  ranking_params: dict[str, float]=None,  
                 drop: tuple[str] = ("ada", "gpc", "lightgbm")):
        # initialize the Trainer class
        super().__init__(model, training_output, num_splits, test_size, outliers, scaler)
        # change the ranking parameters
        ranking_dict = dict(precision_weight=1.2, recall_weight=0.8, report_weight=0.6, 
                            difference_weight=1.2)
        if isinstance(ranking_params, dict):
            for key, value in ranking_params.items():
                if key not in ranking_dict:
                    raise KeyError(f"The key {key} is not found in the ranking params use theses keys: {', '.join(ranking_dict.keys())}")
                ranking_dict[key] = value

        self.experiment.final_models = [x for x in self.experiment.final_models if x not in drop]
        self.pre_weight = ranking_dict["precision_weight"]
        self.rec_weight = ranking_dict["recall_weight"]
        self.report_weight = ranking_dict["report_weight"]
        self.difference_weight = ranking_dict["difference_weight"]
    
    def _calculate_score_dataframe(self, dataframe):
        cv_train = dataframe.loc[("CV-Train", "Mean")]
        cv_val = dataframe.loc[("CV-Val", "Mean")]

        mcc = ((cv_train["MCC"] + cv_val["MCC"])
                - self.difference_weight * abs(cv_val["MCC"] - cv_val["MCC"] ))
        
        prec = ((cv_train["Prec."] + cv_val["Prec."])
                - self.difference_weight * abs(cv_val["Prec."] - cv_train["Prec."]))
        
        recall = ((cv_train["Recall"] + cv_val["Recall"])
                - self.difference_weight * abs(cv_val["Recall"] - cv_train["Recall"]))
        
        return mcc + self.report_weight * (self.pre_weight * prec + self.rec_weight * recall)
    
    def run_holdout(self, feature: DataParser, plot: tuple[str, ...]=("learning", "confusion_matrix", "class_report")):
        """
        A function that splits the data into training and test sets and then trains the models
        using cross-validation but only on the training data

        Parameters
        ----------
        feature : DataParser
            A class containing the training samples and the features
        plot : tuple[str, ...], optional
            Plot the plots relevant to the models, by default 1, 4 and 5
                1. learning: learning curve
                2. pr: Precision recall curve
                3. auc: the ROC curve
                4. confusion_matrix 
                5. class_report: read classification_report from sklearn.metrics

        Returns
        -------
        pd.DataFrame
            A dictionary with the sorted results from pycaret
        list[models]
            A dictionary with the sorted models from pycaret
        pd.DataFrame
         
        """
        self.log.info("------ Running holdout -----")
        X_train, X_test = train_test_split(feature.features, test_size=self.test_size, random_state=self.experiment.seed, 
                                           stratify=feature.features[feature.label])
        sorted_results, sorted_models, top_params = self.setup_holdout(X_train, X_test, self._calculate_score_dataframe, plot)
        return sorted_results, sorted_models, top_params

    def run_kfold(self, feature: DataParser, plot=()):
        """
        A function that splits the data into kfolds of training and test sets and then trains the models
        using cross-validation but only on the training data. It is a nested cross-validation

        Parameters
        ----------
        feature : pd.DataFrame
            A dataframe containing the training samples and the features
        plot : bool, optional
            Plot the plots relevant to the models, by default None
                1. learning: learning curve
                2. pr: Precision recall curve
                3. auc: the ROC curve
                4. confusion_matrix 
                5. class_report: read classification_report from sklearn.metrics

        Returns
        -------
        dict[tuple(dict[pd.DataFrame], dict[models]))]
            A dictionary with the sorted results and sorted models from pycaret organized by split index or kfold index
        """
        self.log.info("------ Running kfold -----")
        skf = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=self.experiment.seed)
        sorted_results, sorted_models, top_params = self.setup_kfold(feature.features, feature.label, skf, plot, feature.with_split)
        return sorted_results, sorted_models, top_params
    