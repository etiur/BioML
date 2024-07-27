import pandas as pd
import pytest
from BioML.utilities.utils import scale, Log, Threshold, read_outlier_file

import numpy as np
import logging


@pytest.fixture
def sample_data():
    # Create sample data for testing
    X_train = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    X_test = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
    return X_train, X_test

def test_scale(sample_data):
    X_train, X_test = sample_data
    transformed, scaler_dict, test_x = scale("robust", X_train, X_test, True)
    transformed2, scaler_dict = scale("zscore", X_train, to_dataframe=False)

    # Check if the scaler object used is RobustScaler
    assert len(set(list(scaler_dict.keys())).intersection(["robust", "zscore", "minmax"])) == 0 
    # Check if the transformed data is a pandas DataFrame
    assert isinstance(transformed, pd.DataFrame)
    assert isinstance(test_x, pd.DataFrame)
    assert isinstance(transformed2, np.ndarray)
    
    # Check if the transformed data has the same shape as the input data
    assert transformed.shape == X_train.shape


@pytest.fixture
def log():
    return Log("test_log")


def test_log_debug(log, caplog):
    log.debug("Debug message")
    assert "Debug message" in caplog.text
    assert caplog.records[-1].levelno == logging.DEBUG


def test_log_info(log, caplog):
    log.info("Info message")
    assert "Info message" in caplog.text
    assert caplog.records[-1].levelno == logging.INFO


def test_log_warning(log, caplog):
    log.warning("Warning message")
    assert "Warning message" in caplog.text
    assert caplog.records[-1].levelno == logging.WARNING


def test_log_error(log, caplog):
    log.error("Error message")
    assert "Error message" in caplog.text
    assert caplog.records[-1].levelno == logging.ERROR


def test_log_critical(log, caplog):
    log.critical("Critical message")
    assert "Critical message" in caplog.text
    assert caplog.records[-1].levelno == logging.CRITICAL


def test_threshold_apply_threshold():
    # Create sample data
    data = pd.DataFrame({'temperature': [25, 30, 35, 40, 45]})

    # Create Threshold object
    threshold = Threshold(data, 'data/data_threshold.csv')

    # Apply threshold
    filtered_data = threshold.apply_threshold(35, "temperature")

    # Check if the filtered data has the expected values
    expected_data = pd.Series([0, 0, 1, 1, 1])
    assert filtered_data.equals(expected_data)


def test_read_outlier_file(tmp_path):
    # Create a temporary file with outliers
    outliers = ("outlier1", "outlier2", "outlier3")
    file_path = tmp_path / "outliers.txt"
    with open(file_path, "w") as f:
        f.write("\n".join(outliers))

    # Test reading outliers from file
    result = read_outlier_file(file_path)
    assert result == outliers

    # Test reading outliers from tuple
    result = read_outlier_file(outliers)
    assert result == outliers

    # Test reading outliers from non-existent file
    result = read_outlier_file("non_existent_file.txt")
    assert result is None