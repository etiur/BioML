import pytest
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from .context import scale, Classifier, Regressor
from collections import defaultdict
from pathlib import Path


class TestRegressor:
    @pytest.fixture
    def data_re(self):
        X, Y = make_regression(n_samples=100, n_features=20, n_informative=10, noise=1, 
                               random_state=42)
        X_data = pd.DataFrame(X)
        Y_data = pd.Series(Y)
        X_data.iloc[:,0] = 0
        Y_data.name = "target"
        return X, Y, X_data, Y_data
    
    @pytest.fixture
    def trainer_re(self, data_re):
        X, Y, X_data, Y_data = data_re
        return Regressor(X_data, Y_data, "../data/training_output", seed=200)

    def test_fix_feature_label_re(self, data_re):
        X, Y, X_data, Y_data = data_re
        csv_file = Regressor(X, Y)
        pandas = Regressor(X_data, Y_data)
        excel_file = Regressor("../data/regression_selected.xlsx", Y)



class TestClassifier:
    
    @pytest.fixture
    def data_cla():
        X, Y = make_classification(n_samples=100, n_features=20, n_informative=10, n_redundant=2, 
                                n_repeated=0, n_classes=2)
        X_data = pd.DataFrame(X)
        Y_data = pd.Series(Y)
        X_data.iloc[:,0] = 0
        Y_data.name = "target"
        return X, Y, X_data, Y_data
   
    @pytest.fixture
    def trainer_cla(self, data_cla):
        X, Y = data_cla
        return Classifier(X, Y, "../data/training_output", seed=200)
    
    def test_fix_feature_label(self, data_cla):
        X, Y, X_data, Y_data = data_cla
        csv_file = Classifier(X, Y)
        pandas = Classifier(X_data, Y_data)
        excel_file = Classifier("../data/classification_selected.xlsx", Y)
