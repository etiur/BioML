import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from BioML.utilities.utils import scale, Log
from BioML.utilities.utils import run_program_subprocess


@pytest.fixture
def sample_data():
    # Create sample data for testing
    X_train = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    X_test = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
    return X_train, X_test

def test_scale_robust(sample_data):
    X_train, X_test = sample_data
    transformed, scaler_dict = scale("robust", X_train, X_test)
    
    # Check if the scaler object used is RobustScaler
    assert isinstance(scaler_dict["robust"], RobustScaler)
    
    # Check if the transformed data is a pandas DataFrame
    assert isinstance(transformed, pd.DataFrame)
    
    # Check if the transformed data has the same shape as the input data
    assert transformed.shape == X_train.shape
    
    # Check if the transformed test data is None
    assert X_test is None

def test_scale_zscore(sample_data):
    X_train, X_test = sample_data
    transformed, scaler_dict, test_x = scale("zscore", X_train, X_test)
    
    # Check if the scaler object used is StandardScaler
    assert isinstance(scaler_dict["zscore"], StandardScaler)
    
    # Check if the transformed data is a pandas DataFrame
    assert isinstance(transformed, pd.DataFrame)
    
    # Check if the transformed data has the same shape as the input data
    assert transformed.shape == X_train.shape
    
    # Check if the transformed test data is a pandas DataFrame
    assert isinstance(test_x, pd.DataFrame)
    
    # Check if the transformed test data has the same shape as the input test data
    assert test_x.shape == X_test.shape

def test_scale_minmax(sample_data):
    X_train, X_test = sample_data
    transformed, scaler_dict, test_x = scale("minmax", X_train, X_test)
    
    # Check if the scaler object used is MinMaxScaler
    assert isinstance(scaler_dict["minmax"], MinMaxScaler)
    
    # Check if the transformed data is a pandas DataFrame
    assert isinstance(transformed, pd.DataFrame)
    
    # Check if the transformed data has the same shape as the input data
    assert transformed.shape == X_train.shape
    
    # Check if the transformed test data is a pandas DataFrame
    assert isinstance(test_x, pd.DataFrame)
    
    # Check if the transformed test data has the same shape as the input test data
    assert test_x.shape == X_test.shape


def test_run_program_subprocess_single_command():
    command = "echo 'Hello, World!'"
    output = run_program_subprocess(command)
    assert output == "Hello, World!\n"

def test_run_program_subprocess_multiple_commands():
    commands = ["echo 'Hello'", "echo 'World!'"]
    output = run_program_subprocess(commands)
    assert output == "Hello\nWorld!\n"

def test_run_program_subprocess_with_program_name():
    command = "echo 'Hello, World!'"
    program_name = "Test Program"
    output = run_program_subprocess(command, program_name=program_name)
    assert output == "Hello, World!\n"

def test_run_program_subprocess_with_shell():
    command = "echo $HOME"
    output = run_program_subprocess(command, shell=True)
    assert output == "/home/user\n"

def test_run_program_subprocess_with_errors():
    command = "invalid_command"
    with pytest.raises(Exception):
        run_program_subprocess(command)

def test_run_program_subprocess_with_output_file():
    command = "echo 'Hello, World!'"
    run_program_subprocess(command)
    with open("output_file.txt", "r") as f:
        output = f.read()
    assert output == "Hello, World!\n"

def test_run_program_subprocess_with_error_file():
    command = "invalid_command"
    with pytest.raises(Exception):
        run_program_subprocess(command)
    with open("error_file.txt", "r") as f:
        error = f.read()
    assert "invalid_command: command not found" in error


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