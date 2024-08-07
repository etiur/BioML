import pytest
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from .context import features, scale
from sklearn.model_selection import train_test_split
from collections import defaultdict
from pathlib import Path


class TestFeatureRegression:
    @pytest.fixture
    def data_regression(self):
        X, Y = make_regression(n_samples=100, n_features=20, n_informative=10, noise=1, 
                               random_state=42)
        X = pd.DataFrame(X)
        Y = pd.Series(Y)
        X.iloc[:,0] = 0
        Y.name = "target"
        return X, Y
    
    @pytest.fixture
    def selection_regressor(self, data_regression):
        X, Y = data_regression
        return features.FeatureRegression(Y, "../data/regression_selected.xlsx", X, seed=200)
    
    def test_read_features_labels_csv(self, selector, data_regression):
        X, Y = data_regression
        features = "../data/regression.csv"
        labels = "../data/regression_labels.csv"
        fixed_features, fixed_labels = selector._read_features_labels(features, labels)
        assert X.equals(fixed_features), "regression csv failed" 
        assert Y.equals(fixed_labels), "regression labels failed"

    def test_read_features_labels_dataframe(self, selector):
        features = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
        labels = pd.Series([0, 1, 0])
        fixed_features, fixed_labels = selector._fix_features_labels(features, labels)
        assert isinstance(fixed_features, pd.DataFrame)
        assert isinstance(fixed_labels, pd.Series)

    def test_read_features_labels_array(self, selector):
        features = [[1, 2, 3], [4, 5, 6]]
        labels = [0, 1]
        fixed_features, fixed_labels = selector._fix_features_labels(features, labels)
        assert isinstance(fixed_features, pd.DataFrame)
        assert isinstance(fixed_labels, pd.Series)

    def test_read_features_labels_invalid(self, selector):
        features = "invalid_features"
        labels = "invalid_labels"
        with pytest.raises(TypeError):
            selector._fix_features_labels(features, labels)

    def test_read_features_labels_label_in_features(self, selector):
        features = pd.DataFrame({"feat1": [1, 2, 3], "target": [0, 1, 0]})
        labels = "target"
        fixed_features, fixed_labels = selector._fix_features_labels(features, labels)
        assert isinstance(fixed_features, pd.DataFrame)
        assert isinstance(fixed_labels, pd.Series)
        assert "target" not in fixed_features.columns
        assert fixed_labels.equals(features["target"])

    def test_fix_features_labels_csv_label_in_features(self, selector):
        features = "../data/target_in_csv_regression.csv"
        labels = "target"
        fixed_features, fixed_labels = selector._fix_features_labels(features, labels)
        assert isinstance(fixed_features, pd.DataFrame)
        assert isinstance(fixed_labels, pd.Series)
        assert "target" not in fixed_features.columns
        assert fixed_labels.equals(fixed_features["target"])

    def test_preprocess(self, selection_regressor, data_regression):
        preprocessed_data = selection_regressor.preprocess()
        assert preprocessed_data.shape == (data_regression.shape[0], data_regression.shape[1]-1)
    
    def test_filter_names_regression(self, selection_regressor):
        filter_args = selection_regressor.filter_args
        assert filter_args["filter_names"] == ['KendallCorr', 'SpearmanCorr', 'PearsonCorr', 'FechnerCorr']
        assert not filter_args["multivariate"]
        assert not filter_args["filter_unsupervised"]
        assert filter_args["regression_filters"] == ("mutual_info", "Fscore")
                
    def test_regressor_hold_out(self, selection_regressor):
        plot=True
        feature_dict = defaultdict(dict)
        num_feature_range = selection_regressor._get_num_feature_range()
        X_train, X_test, Y_train, Y_test = train_test_split(features, selection_regressor.label, test_size=0.20, random_state=selection_regressor.seed)
        transformed, scaler_dict = scale(selection_regressor.scaler, X_train)
        ordered_features = selection_regressor.parallel_filter(transformed, Y_train, num_feature_range[-1], selection_regressor.features.columns, 0,
                                                problem="regression", filter_args=selection_regressor.filter_args, plot=plot)
        
        selection_regressor._construct_features(ordered_features, selection_regressor.features, feature_dict, num_feature_range, transformed, Y_train, 
                                     0, 30, problem="regression")
        
        selection_regressor._write_dict(feature_dict)

        assert num_feature_range == [10, 20, 30, 40, 50], "The feature range function in regression failed"
        assert X_train.shape == (80, 19) and X_test == (20, 19), "The train test split function in regression failed"
        assert transformed.shape == X_train.shape, "The scale function in regression failed"
        assert len(ordered_features.index.unique(0)) == 8, "The parallel filter function in regression failed"
        assert len(feature_dict.keys()) == 45, "The construct features function in regression failed"
        assert Path(selection_regressor.excel_file).exists(), "The write dict function in regression failed"
        if plot:
            files = Path(selection_regressor.excel_file.parent / "shap_features").glob("*")
            assert len(list(files)) > 0, "The plotting of shap in regression failed"
        
        loaded_features = pd.read_excel(selection_regressor.excel_file, sheet_name=list(feature_dict.keys()), index_col=0, header=[0,1])
        assert list(loaded_features.keys()) == list(feature_dict.keys()), "The execel file is different than the generated feature dict"



class TestFeatureClassification:
    
    @pytest.fixture
    def data_classification(self):
        X, Y = make_classification(n_samples=100, n_features=20, n_informative=10, n_redundant=2, 
                                n_repeated=0, n_classes=2)
        X = pd.DataFrame(X)
        Y = pd.Series(Y)
        Y.name = "label"
        return X, Y
    
    @pytest.fixture
    def selection_classification(self, data_classification):
        X, Y = data_classification
        return features.FeatureClassification(Y, "../data/classification_selected.xlsx", X, seed=200)
    
    @pytest.mark.slow
    def test_classification_kfold(self, selection_classification):
        plot=False
        selection_classification.construct_kfold_classification(plot=plot)
        assert Path(selection_classification.excel_file).exists()
    
        with pd.ExcelFile(selection_classification.excel_file) as file:
            assert len(file.sheet_names) == 80
            loaded_features = pd.read_excel(selection_classification.excel_file, sheet_name=file.sheet_names[0], index_col=0, header=[0,1])
        
        if plot:
            files = Path(selection_classification.excel_file.parent / "shap_features").glob("*")
            assert len(list(files)) > 0
        assert len(loaded_features.columns.unique(0)) == selection_classification.num_splits, "kfold classification failed"
    
    def test_filter_names_classification(self, selection_classification):
        filter_args = selection_classification.filter_args
        assert filter_args["filter_names"] == ['FechnerCorr', 'LaplacianScore', 'KendallCorr', 'FRatio', 'SpearmanCorr', 'InformationGain', 'Anova', 
                                               'PearsonCorr', 'SymmetricUncertainty', 'Chi2']
        assert filter_args["multivariate"] == ('STIR', 'TraceRatioFisher')
        assert filter_args["filter_unsupervised"] == ('TraceRatioLaplacian',)
        assert not filter_args["regression_filters"]
        selection_classification.filter_args = ("regression_filters", ("mutual_info"))
        assert selection_classification.filter_args["regression_filters"] == ("mutual_info"), "the property filter_args is not working"


if __name__ == "__main__":
    pytest.main()