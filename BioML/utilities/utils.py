from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import pandas as pd
from pathlib import Path
import logging
from collections import Counter
import shlex
from subprocess import Popen, PIPE
import time
from collections import defaultdict
from typing import Generator, Callable, Iterable, Iterator
import random
import numpy as np
import torch
import json
import yaml
from typing import Any, Callable
from transformers import PreTrainedModel
from dataclasses import dataclass


@dataclass(slots=True)
class FileParser:
    file_path: str | Path

    def load(self, extension: str="json") -> dict[str, Any]:
        with open(self.file_path) as file:
            if extension == "json":
                with open(self.file_path) as file:
                    return json.load(file)
            elif extension == "yaml":
                return yaml.load(file, Loader=yaml.FullLoader)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")

class Log:
    """
    A class to keep log of the output from different modules
    """

    def __init__(self, name: str, level: str="debug"):
        """
        Initialize the Log class

        Parameters
        __________
        name: str
            The name of the log file
        level: str, default="debug"
            The logging level. Options are: "debug", "info", "warning", "error", "critical"
        """
        level_ = {"debug": logging.DEBUG, 
                 "info": logging.INFO, 
                 "warning": logging.WARNING, 
                 "error": logging.ERROR, 
                 "critical": logging.CRITICAL}
        
        self._logger = logging.getLogger(name)
        self._logger.handlers = []
        self._logger.setLevel(level_[level])
        self.fh = logging.FileHandler(f"{name}.log")
        self.fh.setLevel(level_[level])
        self.ch = logging.StreamHandler()
        self.ch.setLevel(level_[level])
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', "%d-%m-%Y %H:%M:%S")
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        self._logger.addHandler(self.fh)
        self._logger.addHandler(self.ch)

    def debug(self, messages: str, exc_info: bool=False):
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

    def info(self, messages: str, exc_info: bool=False):
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

    def warning(self, messages: str, exc_info: bool=False):
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

    def error(self, messages: str, exc_info: bool=False):
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

    def critical(self, messages: str, exc_info: bool=False):
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

    def save_csv(self, threshold: int | float, greater: bool=True, column_name: str='temperature'):
        data = self.apply_threshold(threshold, greater, column_name)
        data.to_csv(self.output_csv)


class MmseqsClustering:
    @classmethod
    def create_database(cls, input_file: str | Path, output_database: str):
        """
        Create a database from a fasta file.

        Parameters
        ----------
        input_file : str | Path
            The path to the input fasta file.
        output_database : str
            The path to the output database.
        """
        input_file = Path(input_file)
        output_index = Path(output_database)
        output_index.parent.mkdir(exist_ok=True, parents=True)
        command = f"mmseqs createdb {input_file} {output_index}"
        run_program_subprocess(command, "createdb")

    @classmethod
    def index_database(cls, database: str | Path):
        """
        Index the target database if it is going to be reused for search 
        frequently. This will speed up the search process because it loads in memory.

        Parameters
        ----------
        database : str | Path
            The path to the database.
        """
        database = Path(database)
        command = f"mmseqs createindex {database} tmp"
        run_program_subprocess(command, "create index")
    
    @classmethod
    def cluster(cls, database: str | Path, cluster_tsv: str="cluster.tsv", 
                cluster_at_sequence_identity: float =0.3, 
                sensitivity: float=6.5, **cluster_kwargs: dict):
        
        """
        Cluster sequences in the given database using MMseqs2.

        Parameters
        ----------
        database : str or Path
            Path to the input sequence database.
        cluster_tsv : str or Path, optional
            Path to the output cluster TSV file, by default "cluster.tsv".
        cluster_at_sequence_identity : float, optional
            Sequence identity threshold for clustering, by default 0.3.
        sensitivity : float, optional
            Sensitivity parameter for MMseqs2, by default 6.5.
        **cluster_kwargs : dict, optional
            Additional keyword arguments to pass to the MMseqs2 cluster command.

        Returns
        -------
        None

        Raises
        ------
        SubprocessError
            If the MMseqs2 commands fail.

        Notes
        -----
        This method runs two MMseqs2 commands: 'cluster' and 'createtsv'.
        The 'cluster' command performs the actual clustering of sequences,
        and the 'createtsv' command creates a TSV file with the clustering results.
        """
        database = Path(database)
        intermediate_output = Path("cluster_output/clusterdb")
        intermediate_output.parent.mkdir(exist_ok=True, parents=True)
        output_cluster = Path(cluster_tsv)
        output_cluster.parent.mkdir(exist_ok=True, parents=True)
        cluster = f"mmseqs cluster {database} {intermediate_output} tmp --min-seq-id {cluster_at_sequence_identity} --cluster-reassign --alignment-mode 3 -s {sensitivity}"
        for key, value in cluster_kwargs.items():
            cluster += f" --{key} {value}"
        createtsv = f"mmseqs createtsv {database} {database} {intermediate_output} {output_cluster}"
        run_program_subprocess(cluster, "cluster")
        run_program_subprocess(createtsv, "create tsv")
    
    @classmethod
    def generate_pssm(cls, query_db: str | Path, search_db: str | Path, 
                      evalue: float=0.01, num_iterations: int=3, pssm_filename: str="result.pssm", 
                      max_seqs: int=600, sensitivity: float=6.5, **search_kwags: dict):
        """
        Generate a Position-Specific Scoring Matrix (PSSM) using MMseqs2.

        Parameters
        ----------
        query_db : str or Path
            Path to the query sequence database.
        search_db : str or Path
            Path to the search sequence database.
        evalue : float, optional
            E-value threshold for the search, by default 0.01.
        num_iterations : int, optional
            Number of search iterations, by default 3.
        pssm_filename : str or Path, optional
            Path to the output PSSM file, by default "result.pssm".
        max_seqs : int, optional
            Maximum number of sequences to keep per query, by default 600.
        sensitivity : float, optional
            Sensitivity parameter for MMseqs2, by default 6.5.
        **search_kwags : dict, optional
            Additional keyword arguments to pass to the MMseqs2 search command.

        Returns
        -------
        None

        Raises
        ------
        SubprocessError
            If the MMseqs2 commands fail.

        Notes
        -----
        This method runs three MMseqs2 commands: 'search', 'result2profile', and 'profile2pssm'.
        The 'search' command performs the sequence search,
        the 'result2profile' command generates a profile from the search results,
        and the 'profile2pssm' command converts the profile to a PSSM.
        """
        search = f"mmseqs search {query_db} {search_db} result.out tmp -e {evalue} --num-iterations {num_iterations} --max-seqs {max_seqs} -s {sensitivity} -a"
        for key, value in search_kwags.items():
            search += f" --{key} {value}"
        run_program_subprocess(search, "search")
        profile = f"mmseqs result2profile {query_db} {search_db} result.out result.profile"
        run_program_subprocess(profile, "generate_profile")
        pssm = f"mmseqs profile2pssm result.profile {pssm_filename}"
        run_program_subprocess(pssm, "convert profile to pssm")
    
    @classmethod
    def easy_cluster(cls, input_file: str | Path, cluster_tsv: str | Path, 
                    cluster_at_sequence_identity: float = 0.3, sensitivity: float = 6.5, 
                    **cluster_kwargs: dict):
        """
        Easily cluster sequences in the given input file using MMseqs2.

        Parameters
        ----------
        input_file : str or Path
            Path to the input sequence file.
        cluster_tsv : str or Path
            Path to the output cluster TSV file.
        cluster_at_sequence_identity : float, optional
            Sequence identity threshold for clustering, by default 0.3.
        sensitivity : float, optional
            Sensitivity parameter for MMseqs2, by default 6.5.
        **cluster_kwargs : dict, optional
            Additional keyword arguments to pass to the MMseqs2 cluster command.

        Returns
        -------
        dict
            Dictionary with cluster information.

        Notes
        -----
        This method creates a database from the input file, clusters the sequences,
        and reads the cluster information.
        """
        query_db = Path(input_file).with_suffix("")/"querydb"
        if not query_db.exists():
            cls.create_database(input_file, query_db)
        cls.cluster(query_db, cluster_tsv, cluster_at_sequence_identity, 
                    sensitivity, **cluster_kwargs)
        return cls.read_cluster_info(cluster_tsv)
    
    @classmethod
    def read_cluster_info(cls, file_path: str | Path):
        """
        Read cluster information from a file.

        Parameters
        ----------
        file_path : str or Path
            Path to the file with cluster information.

        Returns
        -------
        dict
            Dictionary with cluster information.
        """
        cluster_info = {}
        with open(file_path, "r") as f:
            lines = [x.strip() for x in f.readlines()]
        for x in lines:
            X = x.split("\t")
            if X[0] not in cluster_info:
                cluster_info[X[0]] = []
            cluster_info[X[0]].append(X[1])
        return cluster_info

    @classmethod
    def easy_generate_pssm(cls, input_file: str | Path, database_file: str | Path, 
                        evalue: float = 0.01, num_iterations: int = 3, 
                        sensitivity: float = 6.5, pssm_filename: str = "result.pssm", 
                        generate_searchdb: bool = False, max_seqs: int = 600, 
                        **search_kwags: dict):
        """
        Easily generate a Position-Specific Scoring Matrix (PSSM) using MMseqs2.

        Parameters
        ----------
        input_file : str or Path
            Path to the input sequence file.
        database_file : str or Path
            Path to the database file.
        evalue : float, optional
            E-value threshold for the search, by default 0.01.
        num_iterations : int, optional
            Number of search iterations, by default 3.
        sensitivity : float, optional
            Sensitivity parameter for MMseqs2, by default 6.5.
        pssm_filename : str or Path, optional
            Path to the output PSSM file, by default "result.pssm".
        generate_searchdb : bool, optional
            Whether to generate a search database from the database file, by default False.
        max_seqs : int, optional
            Maximum number of sequences to keep per query, by default 600.
        **search_kwags : dict, optional
            Additional keyword arguments to pass to the MMseqs2 search command.

        Returns
        -------
        str
            Path to the generated PSSM file.

        Notes
        -----
        This method creates databases from the input and database files if they do not exist,
        and generates a PSSM from the query and search databases.
        """

        query_db = Path(input_file).with_suffix("")/"querydb"
        search_db = Path(database_file)
        # generate the databases using the fasta files from input and the search databse like uniref
        if not query_db.exists():
            cls.create_database(input_file, query_db)
        if generate_searchdb:
            search_db = search_db.with_suffix("")/"searchdb"
            cls.create_database(database_file, search_db)

        # generate the pssm files
        cls.generate_pssm(query_db, search_db, evalue, num_iterations, pssm_filename, 
                          max_seqs, sensitivity, **search_kwags)
        return pssm_filename
    
    @classmethod
    def split_pssm(cls, pssm_filename: str | Path, output_dir: str | Path ="pssm"):
        cls.write_pssm(cls.iterate_pssm(pssm_filename), output_dir)

    @classmethod
    def iterate_pssm(cls, pssm_filename: str | Path) -> Generator[tuple[int, list[str]], None, None]:
        pssm_dict = defaultdict(list)
        current_seq = None
        with open(pssm_filename, "r") as f:
            for x in f:
                if x.startswith('Query profile of sequence'):
                    seq = int(x.split(" ")[-1])
                    if current_seq is not None and seq != current_seq:
                        yield current_seq, pssm_dict[current_seq]
                        del pssm_dict[current_seq]
                    current_seq = seq
                pssm_dict[seq].append(x)
        if current_seq is not None:
            yield current_seq, pssm_dict[current_seq] 

    @classmethod
    def write_pssm(cls, pssm_tuple: Generator[tuple[int, list[str]], None, None], 
                   output_dir: str | Path ="pssm"):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        for key, value in pssm_tuple:
            hold = ["\n"]
            hold.extend(value)
            with open(f"{output_dir}/pssm_{key}.pssm", "w") as f:
                f.writelines(hold)

def set_seed(seed: int):
    """
    Set the seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        The seed value to set.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True # use deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    
def convert_to_parquet(csv_file: str | Path, parquet_file: str | Path):
    """
    Convert a CSV file to parquet format.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file.
    parquet_file : str
        Path to the parquet file.
    """
    df = pd.read_csv(csv_file, index_col=0)
    df.to_parquet(parquet_file)
    Path(csv_file).unlink()


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
    #transformed = pd.DataFrame(transformed, index=X_train.index, columns=X_train.columns)
    if X_test is None:
        return transformed, scaler_dict
    
    test_x = scaler_dict[scaler].transform(X_test)
    #test_x = pd.DataFrame(test_x, index=X_test.index, columns=X_test.columns)
    return transformed, scaler_dict, test_x


def read_outlier_file(outliers: tuple[str,...] | str | None=None) -> tuple[str,...] | None:
    """
    Read the outliers from a file.

    Parameters
    ----------
    outliers : tuple[str,...] | str | None, optional
        A tuple containing the outliers or the path to the file containing the outliers.

    Returns
    -------
    tuple[str,...] | None
        A tuple containing the outliers.
    """
    if outliers and Path(outliers[0]).exists():
        with open(outliers) as out:
            outliers = tuple(x.strip() for x in out.readlines())
    return outliers


def write_excel(file: str | pd.io.excel._openpyxl.OpenpyxlWriter, 
                dataframe: pd.DataFrame | pd.Series, sheet_name: str) -> None:
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
        if not Path(file).exists():
            with pd.ExcelWriter(file, mode="w", engine="openpyxl") as writer:
                dataframe.to_excel(writer, sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(file, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                dataframe.to_excel(writer, sheet_name=sheet_name)              
    else:
        dataframe.to_excel(file, sheet_name=sheet_name)


def run_program_subprocess(commands: list[str] | str, program_name: str | None=None, 
                           shell: bool=False):
    """
    Run in parallel the subprocesses from the command
    Parameters
    ----------
    commands: list[str] | str
        A list of commandline commands that calls to Possum programs or ifeature programs
    program_name: str, optional
        A name to identify the commands
    shell: bool, optional
        If running the commands through the shell, allowing you to use shell features like 
        environment variable expansion, wildcard matching, and various other shell-specific features
        If False, the command is expected to be a list of individual command-line arguments, 
        and no shell features are applied. It is safer shell=False when there is user input to avoid
        shell injections.
    """
    if isinstance(commands, str):
        commands = [commands]
    proc = []
    for command in commands:
        if shell:
            proc.append(Popen(command, stderr=PIPE, stdout=PIPE, text=True, shell=shell))
        else:
            proc.append(Popen(shlex.split(command), stderr=PIPE, stdout=PIPE, text=True, shell=shell))

    start = time.time()
    err = []
    out = []
    for p in proc:
        output, errors = p.communicate()
        if output: out.append(output)
        if errors: err.append(errors)
    
    if err:
        with open("error_file.txt", "w") as ou:
            ou.writelines(f"{err}")
    if out:
        with open("output_file.txt", "w") as ou:
            ou.writelines(f"{out}")
            
    end = time.time()
    if program_name:
        print(f"start running {program_name}")
    print(f"It took {end - start} second to run")


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


def evaluate_all_models(evaluation_fn: Callable, results: dict[str, dict[str, tuple | dict]], 
                        training_output: str | Path) -> None:
    """
    Evaluate all models using the given evaluation function and save the results. The function used here plots the learning curve.
    It is easier to extend this function to evaluate it other ways.

    Parameters
    ----------
    evaluation_fn : Callable
        The evaluation function to use for evaluating the models.
    results : dict[str, dict[str, tuple]]
        A dictionary containing the results of the models to be evaluated.
    training_output : str | Path
        The path to the directory where the evaluation results will be saved.

    Returns
    -------
    None
        This function does not return anything, it only saves the evaluation results.
    """
    for tune_status, result_dict in results.items():
        for key, value in result_dict.items():
            if key == "stacked" or key == "majority":
                try:
                    evaluation_fn(value, save=f"{training_output}/evaluation_plots/{tune_status}/{key}")
                except AttributeError:
                    pass
            elif tune_status == "tuned" and key == "holdout":
                for mod_name, model in value.items(): # type: ignore
                    try:
                        evaluation_fn(model, save=f"{training_output}/evaluation_plots/{tune_status}/{key}/{mod_name}")
                    except AttributeError:
                        pass


def write_results(training_output: Path | str, sorted_results: pd.DataFrame, top_params: pd.Series | None = None, 
                  sheet_name: str|None=None) -> None:
    """
    Writes the training results and top hyperparameters to Excel files.

    Parameters
    ----------
    training_output : Path | str
        The path to the directory where the Excel files will be saved.
    sorted_results : pd.DataFrame
        A pandas DataFrame containing the sorted training results.
    top_params : pd.Series or None, optional
        A pandas Series containing the top hyperparameters. Defaults to None.
    sheet_name : str or None, optional
        The name of the sheet to write the results to. Defaults to None.

    Returns
    -------
    None
    """
    training_output = Path(training_output)
    training_output.mkdir(exist_ok=True, parents=True)
    write_excel(training_output / "training_results.xlsx", sorted_results, sheet_name) # type: ignore
    if top_params is not None:
        write_excel(training_output / "top_hyperparameters.xlsx", top_params, sheet_name) # type: ignore


def iterate_multiple_features(iterator: Iterator, parser, label: str | list[int | float], 
                              training, outliers: Iterable[str],
                              training_output: Path, **kwargs: Any) -> None:
    
    """
    Iterates over multiple input features and generates training results for each feature.

    Parameters
    ----------
    iterator : Iterator
        An iterator that yields a tuple of input features and sheet names.
    parser : DataParser
        The data parser to use for parsing the input features.
    label : str or list[int or float]
        The label or list of labels to use for training.
    training : Trainer
        The training object to use for training the model.
    outliers : Iterable[str]
        An iterable containing the names of the outlier detection methods to use for each sheet.
    training_output : Path
        The path to the directory where the training results will be saved.

    Returns
    -------
    None
    """

    performance_list = []
    for input_feature, sheet in iterator:
        feature = parser(input_feature, label=label, sheets=sheet, outliers=outliers)
        sorted_results, sorted_models, top_params = training.run_training(feature.feature, feature.label_name, **kwargs)
        index = sorted_results.index.unique(0)[:training.experiment.best_model]
        score = 0
        for i in index:
            score += training.arguments._calculate_score_dataframe(sorted_results.loc[i])
        performance_list.append((sheet, sorted_results.loc[index], score))
    performance_list.sort(key=lambda x: x[2], reverse=True)
    for sheet, performance, score in performance_list:
        write_results(training_output, performance, sheet_name=sheet)
    

def iterate_excel(excel_file: str | Path, sheet_names: Iterable[str] = ()):
    """
    Iterates over the sheets of an Excel file and yields a tuple of the sheet data and sheet name.

    Parameters
    ----------
    excel_file : str or Path
        The path to the Excel file.
    sheet_names : Iterable[str], optional
        An iterable containing the names of the sheets to iterate over. Defaults to an empty tuple.
    Yields
    ------
    Tuple[pd.DataFrame, str]
        A tuple of the sheet data and sheet name.
    """
    with pd.ExcelFile(excel_file) as file:
        for sheet in file.sheet_names:
            if sheet_names and sheet not in sheet_names: continue
            df = pd.read_excel(excel_file, index_col=0, sheet_name=sheet)
            yield df, sheet


def estimate_deepmodel_size(model: PreTrainedModel, precision: torch.dtype):
    """
    Estimate the size of the model in memory.

    Parameters
    ----------
    model : PreTrainedModel
        The pre-trained model to estimate the size of.
    precision : torch.dtype
        The precision of the model's parameters. Can be torch.float16 for half precision or torch.float32 for single precision.

    Returns
    -------
    str
        The estimated size of the model in megabytes (MB), rounded to two decimal places.
    """
    num = 2 if precision==torch.float16 else 4 # float16 takes 2 bytes and float32 takes 4 bytes per parameter
    size = round(model.num_parameters() * num/1000_000, 2)
    return f"{size} MB"


def print_trainable_parameters(model: PreTrainedModel):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

