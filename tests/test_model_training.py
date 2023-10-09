import pytest
import pandas as pd
from sklearn.datasets import make_classification, make_regression

class TestRegressor:
    @pytest.fixture
    def data_regression():
        X, Y = make_regression(n_samples=100, n_features=20, n_informative=10, noise=1, 
                               random_state=42)
        X_data = pd.DataFrame(X)
        Y_data = pd.Series(Y)
        X_data.iloc[:,0] = 0
        Y_data.name = "target"
        return X, Y, X_data, Y_data
    
    @pytest.fixture
    def excel_regression():
        X = pd.read_excel("../data/regression_selected.xlsx", index_col=0, header=[0,1])
        return X, "../data/regression_selected.xlsx"

    def test_fix_feature_label(data_regression, excel_regression):
        X, Y, X_data, Y_data = data_regression
        X_excel, excel_file = excel_regression
        





class TestClassifier:
    
    @pytest.fixture
    def data_classification():
        X, Y = make_classification(n_samples=100, n_features=20, n_informative=10, n_redundant=2, 
                                n_repeated=0, n_classes=2)
        X_data = pd.DataFrame(X)
        Y_data = pd.Series(Y)
        X_data.iloc[:,0] = 0
        Y_data.name = "target"
        return X, Y, X_data, Y_data
    
    @pytest.fixture
    def excel_classification():
        X = pd.read_excel("../data/classification_selected.xlsx", index_col=0, header=[0,1])
        return X, "../data/classification_selected.xlsx"