from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import pandas as pd
from pathlib import Path
import logging
from collections import Counter


def scale(scaler: str, X_train: pd.DataFrame, 
          X_test: pd.DataFrame | None=None) -> tuple[pd.DataFrame, ...]:
    """
    Scale the features using RobustScaler, StandardScaler or MinMaxScaler.

    Parameters
    ----------
    scaler : str
        The type of scaler to use. Must be one of "robust", "zscore", or "minmax".
    X_train : pandas DataFrame object
        The training data.
    X_test : pandas DataFrame object, default=None
        The test data.

    Returns
    -------
    tuple
        A tuple containing:
        - transformed : pandas DataFrame object
            The transformed training data.
        - scaler_dict : dictionary
            A dictionary containing the scaler object used for scaling.
        - test_x : pandas DataFrame object, default=None
            The transformed test data.

    Notes
    -----
    This function scales the features using RobustScaler, StandardScaler or MinMaxScaler. 
    The function first creates a dictionary containing the scaler objects for each type of scaler. 
    It then applies the selected scaler to the training data and returns the transformed data as a pandas DataFrame object. 
    If test data is provided, the function applies the same scaler to the test data and returns the transformed test data as a pandas DataFrame object. 
    The function also returns a dictionary containing the scaler object used for scaling.

    """
    scaler_dict = {"robust": RobustScaler(), "zscore": StandardScaler(), "minmax": MinMaxScaler()}
    transformed = scaler_dict[scaler].fit_transform(X_train)
    transformed = pd.DataFrame(transformed, index=X_train.index, columns=X_train.columns)
    if X_test is None:
        return transformed, scaler_dict
    else:
        test_x = scaler_dict[scaler].transform(X_test)
        test_x = pd.DataFrame(test_x, index=X_test.index, columns=X_test.columns)
        return transformed, scaler_dict, test_x


def write_excel(file, dataframe: pd.DataFrame | pd.Series, sheet_name: str, overwrite: bool=False) -> None:
    """
    Write a pandas DataFrame to an Excel file.

    Parameters
    ----------
    file : str or pandas ExcelWriter object
        The file path or ExcelWriter object to write to.
    dataframe : pandas DataFrame object
        The DataFrame to write to the Excel file.
    sheet_name : str
        The name of the sheet to write to.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None

    Notes
    -----
    This function writes a pandas DataFrame to an Excel file. If the file does not exist, it creates a new file. 
    If the file exists and `overwrite` is set to `True`, it overwrites the file. 
    If the file exists and `overwrite` is set to `False`, it appends the DataFrame to the existing file. 
    The function uses the `openpyxl` engine to write to the Excel file.

    Examples
    --------
    >>> from utilities import write_excel
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> write_excel('example.xlsx', df, 'Sheet1')
    """
    if not isinstance(file, pd.io.excel._openpyxl.OpenpyxlWriter):
        if overwrite or not Path(file).exists():
            mode = "w"
        else:
            mode = "a"
        with pd.ExcelWriter(file, mode=mode, engine="openpyxl") as writer:
            dataframe.to_excel(writer, sheet_name=sheet_name)
    else:
        dataframe.to_excel(file, sheet_name=sheet_name)


def rewrite_possum(possum_stand_alone_path: str | Path) -> None:
    """
    Rewrite the Possum standalone file to use the local Possum package.

    Parameters
    ----------
    possum_stand_alone_path : str or Path object
        The path to the Possum standalone file ending in .pl since it is in perl.

    Returns
    -------
    None

    Notes
    -----
    This function rewrites the Possum standalone file to use the local Possum package. 
    It takes in the path to the Possum standalone file as a string or Path object. 
    The function reads the file, replaces the path to the Possum package with the local path, and writes the updated file.

    Examples
    --------
    >>> from utilities import rewrite_possum
    >>> rewrite_possum('possum_standalone.pl')
    """
    possum_path = Path(possum_stand_alone_path)
    with possum_path.open() as possum:
        possum = possum.readlines()
        new_possum = []
        for line in possum:
            if "python" in line:
                new_line = line.split(" ")
                if "possum.py" in line:
                    new_line[2] = f"{possum_path.parent}/src/possum.py"
                else:
                    new_line[2] = f"{possum_path.parent}/src/headerHandler.py"
                line = " ".join(new_line)
                new_possum.append(line)
            else:
                new_possum.append(line)
    with open(possum_path, "w") as possum_out:
        possum_out.writelines(new_possum)

class Log:
    """
    A class to keep log of the output from different modules
    """

    def __init__(self, name):
        """
        Initialize the Log class

        Parameters
        __________
        name: str
            The name of the log file
        """
        self._logger = logging.getLogger(name)
        self._logger.handlers = []
        self._logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler("{}.log".format(name))
        self.fh.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%d-%m-%Y %H:%M:%S")
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        self._logger.addHandler(self.fh)
        self._logger.addHandler(self.ch)

    def debug(self, messages, exc_info=False):
        """
        It pulls a debug message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message            
        """
        self._logger.debug(messages, exc_info=exc_info)

    def info(self, messages, exc_info=False):
        """
        It pulls an info message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message            
        """
        self._logger.info(messages, exc_info=exc_info)

    def warning(self, messages, exc_info=False):
        """
        It pulls a warning message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message            
        """
        self._logger.warning(messages, exc_info=exc_info)

    def error(self, messages, exc_info=False):
        """
        It pulls a error message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message
        """
        self._logger.error(messages, exc_info=exc_info)

    def critical(self, messages, exc_info=False):
        """
        It pulls a critical message.

        Parameters
        ----------
        messages : str
            The messages to log
        exc_info : bool, optional
            Set to true to include the exception error message
        """
        self._logger.critical(messages, exc_info=exc_info)


class Threshold:
    """
    A class to convert regression labels into classification labels.

    Parameters
    ----------
    input_data : pd.DataFrame
        The data to apply the threshold
    output_csv : str or Path object
        The path to the output CSV file.

    Methods
    -------
    apply_threshold(threshold) -> None
        Apply a threshold to the regression labels.
    save_csv() -> None
        Save the CSV file to disk.
    
    Notes
    -----
    This class reads in a CSV file as a pandas DataFrame object and applies a threshold to the regression labels.
    It then saves the DataFrame object as a CSV file.

    Examples
    --------
    >>> from utilities import Threshold
    >>> import pandas as pd
    >>> data = pd.read_csv('data/data.csv')
    >>> threshold = Threshold(data, 'data/data_threshold.csv')
    >>> threshold.apply_threshold(0.5)
    >>> threshold.save_csv()
    """

    def __init__(self, input_data: pd.DataFrame, output_csv: str | Path):
        self.csv_file = input_data
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(exist_ok=True, parents=True)

    def apply_threshold(self, threshold: float, greater: bool=True, column_name: str='temperature'):
        """
        Convert a regression problem to a classification problem by applying a threshold to a column in the dataset.

        Parameters
        ----------
        threshold : float
            The threshold value to apply.
        greater : bool, default=True
            A boolean value to indicate if the threshold is greater or lower.
        column_name : str, default='temperature'
            The name of the column to apply the threshold to.

        Returns
        -------
        pandas Series object
            The filtered dataset.

        Notes
        -----
        This method converts a regression problem to a classification problem by applying a threshold to a column in the dataset. 
        It takes in the threshold value, a boolean value to indicate if the positive class or class 1 should be greater or lower, 
        and the name of the column to apply the threshold to. 
        The method first creates a copy of the column and converts it to a numeric data type. 
        It then drops any rows with missing values. 
        The method then creates a new copy of the column and applies the threshold to it, setting values above or below the 
        threshold to 1 or 0, respectively. The method returns the filtered dataset as a pandas Series object.

        Examples
        --------
        >>> from utilities import Threshold
        >>> import pandas as pd
        >>> data = pd.read_csv('data/data.csv')
        >>> threshold = Threshold(data, 'data/data_threshold.csv')
        >>> threshold.save_csv()
        """
        dataset = self.csv_file[column_name].copy()
        dataset = pd.to_numeric(dataset, errors="coerce")
        dataset.dropna(inplace=True)
        data = dataset.copy()
        if greater:
            data.loc[dataset >= threshold] = 1
            data.loc[dataset < threshold] = 0
        else:
            data.loc[dataset <= threshold] = 1
            data.loc[dataset > threshold] = 0
        print(f"using the threshold {threshold} returns these proportions", Counter(data))
        return data

    def save_csv(self, threshold, greater=True, column_name='temperature'):
        data = self.apply_threshold(threshold, greater, column_name)
        data.to_csv(self.output_csv)