import argparse
import shutil
from Bio import SeqIO
from Bio.SeqIO import FastaIO
from ..utilities import rewrite_possum, run_program_subprocess
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Sequence, Callable
from functools import partial
from ..training.base import DataParser
from dataclasses import dataclass, field
import pandas as pd


def arg_parse():
    parser = argparse.ArgumentParser(description="extract features using possum and ifeatures")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path", required=True)
    parser.add_argument("-p", "--pssm_dir", help="The pssm files directory's path", required=False,
                        default="pssm")
    parser.add_argument("-id", "--ifeature_dir", required=False, help="Path to the iFeature programme folder",
                        default="iFeature")
    parser.add_argument("-Po", "--possum_dir", required=False, help="A path to the possum programme",
                        default="POSSUM_Toolkit/")
    parser.add_argument("-io", "--ifeature_out", required=False, help="The directory where the ifeature features are",
                        default="ifeature_features")
    parser.add_argument("-po", "--possum_out", required=False, help="The directory for the possum extractions",
                        default="possum_features")
    parser.add_argument("-eo", "--extracted_out", required=False,
                        help="The directory for the extracted features (concatenated from both programs)",
                        default="extracted_features")
    parser.add_argument("-e", "--training_features", required=False,
                        help="The path to where the selected features from training are saved in excel format, it will"
                             "be used to select specific columns from all the generated features for the new data",
                        default="training_features/selected_features.xlsx")
    parser.add_argument("-on", "--purpose", nargs="+", choices=("extract", "read", "filter"),
                        default=("extract", "read"),
                        help="Choose the operation between extracting, reading for training or filtering")
    parser.add_argument("-lg", "--long", required=False, help="true when restarting from the long commands",
                        action="store_true")
    parser.add_argument("-r", "--run", required=False, choices=("possum", "ifeature"), nargs="+", default="both",
                        help="run possum or ifeature extraction, the choice will affect --type argument since only one of the programs would be extracted")
    parser.add_argument("-n", "--num_thread", required=False, default=100, type=int,
                        help="The number of threads to use for the generation of pssm profiles")
    parser.add_argument("-d", "--drop", required=False, default="all", nargs="+", choices=("APAAC", "PAAC",
                        "CKSAAGP", "Moran", "Geary", "NMBroto", "CTDC", "CTDT", "CTDD", "CTriad", "GDPC", "GTPC",
                        "QSOrder", "SOCNumber", "GAAC", "KSCTriad", "aac_pssm", "ab_pssm", "d_fpssm", "dp_pssm",
                        "dpc_pssm", "edp", "eedp", "rpm_pssm", "k_separated_bigrams_pssm", "pssm_ac", "pssm_cc",
                        "pssm_composition", "rpssm", "s_fpssm", "smoothed_pssm:5", "smoothed_pssm:7", "smoothed_pssm:9",
                        "tpc", "tri_gram_pssm", "pse_pssm:1", "pse_pssm:2", "pse_pssm:3"),
                        help="A list of the features to extract")
    parser.add_argument("-df", "--drop_file", required=False, help="the path to the file with the feature names, separated by newlines")
    parser.add_argument("-s", "--sheets", required=False, nargs="+", help="Names or index of the selected sheets from the training features")
    parser.add_argument("-t", "--new_features", required=False, help="The path to the features from the new samples so "
                                                "it can be filtered according to the training features")

    args = parser.parse_args()

    return [args.fasta_file, args.pssm_dir, args.ifeature_dir, args.possum_dir, args.ifeature_out,
            args.possum_out, args.extracted_out, args.purpose, args.long, args.run, args.num_thread, args.drop,
            args.drop_file, args.sheets, args.training_features, args.new_features]


class ExtractFeatures:
    """
    A class to extract features using Possum and iFeatures
    """

    def __init__(self, fasta_file: str | Path):
        """
        Initialize the ExtractFeatures class

        Parameters
        ___________
        fasta_file: str
            The fasta file to be analysed
        """
        self.fasta_file = Path(fasta_file)

    @staticmethod
    def _batch_iterable(iterable: Sequence[str], batch_size: int):
        """
        Create generators from iterable

        Parameters
        ----------
        iterable : list | set | tuple
            An iterable containing the object to be batched
        batch_size : int
            The length of the batch

        Yields
        ------
        generator

        """
        length = len(iterable)
        for ndx in range(0, length, batch_size):
            yield iterable[ndx:min(ndx + batch_size, length)]

    def separate_bunch(self):
        """
        A class that separates the fasta files into smaller fasta files

        """
        if self.fasta_file.with_name("group_1.fasta").exists():
            return
        
        with open(self.fasta_file) as inp:
            record = list(SeqIO.parse(inp, "fasta"))
            if len(record) > 5_000:
                for i, batch in enumerate(self._batch_iterable(record, 5_000)):
                    filename = f"group_{i+1}.fasta"
                    with open(self.fasta_file.with_name(filename), "w") as split:
                        print(self.fasta_file.with_name(filename))
                        fasta_out = FastaIO.FastaWriter(split, wrap=None)
                        fasta_out.write_file(batch)
                del record
            else:
                shutil.copyfile(self.fasta_file, f"{self.fasta_file.parent}/group_1.fasta")

    def run_extraction_parallel(self, file: list[str|Path], num_thread: int, 
                                *run: Callable[[str|Path], None]):
        """
        Using a pool of workers to run the 2 programmes

        Parameters
        ----------
        file : list
            A list of the files to be processed
        long:
            If to run only the longer features
        """
        with get_context("spawn").Pool(processes=num_thread) as pool:
            for func in run:
                pool.map(func, file)


@dataclass(slots=True)
class PossumFeatures:
    """
    A class used to represent the features of a Possum.

    Attributes
    ----------
    pssm_dir : str | Path
        The directory where PSSM files are stored. Default is "pssm".
    output : str | Path
        The output directory where the results will be stored. Default is "possum_features".
    program : str | Path
        The program used for the Possum Toolkit. Default is "POSSUM_Toolkit".
    features : dict
        A dictionary to store the features of the Possum. Default is an empty dictionary.
    drop_file : str | Path | None
        The file to be dropped. Default is None.
    drop : Iterable[str]
        An iterable containing the features to be dropped. Default is an empty tuple.
    """
    pssm_dir: str | Path = "pssm"
    output: str | Path = "possum_features"
    program: str | Path = "POSSUM_Toolkit"
    drop_file: str | Path | None = None
    drop: Iterable[str] = ()
    features: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.program = f"{self.program}/possum_standalone.pl"
        rewrite_possum(self.program)
        self.features = return_features("possum", self.drop_file, self.drop)

        for key, feat in self.features.items():
            self.features[key] = list(set(feat).difference(self.drop))
        
        Path(self.output).mkdir(parents=True, exist_ok=True)
        
        print(f"Possum features to be extracted: {self.features}")

    def generate_commands(self, fasta_file: str | Path, programs: list[str]) -> list[str]:
        """
        Writing the commands to run the possum features that uses a different command structure

        Parameters
        ----------
        fasta_file : str | Path
            path to the different fasta files
        programs : list[str]
            list of the different programs to run

        Returns
        -------
        list[str]
            list of the commands to run
        """

        num = Path(fasta_file).stem.split("_")[1]
        command = []
        for prog in programs:
            if ":" in prog:
                name, index = prog.split(":")
                supplement = f'-t {name} -a {index} -o {self.output}/{name}_{index}_{num}.csv'
            else:
                supplement = f'-t {prog} -o {self.output}/{prog}_{num}.csv'

            string = f'perl {self.program} -i {fasta_file} -p {self.pssm_dir} {supplement}'
            command.append(string)

        return command
    
    def extract(self, fasta_file: str | Path, long: bool=False):
        """
        Extracting the features  using Possum

        Parameters
        ----------
        fasta_file : str | Path
            path to the different fasta files
        long : bool, optional
            To only extract long features or not, by default False
        """
        command = self.generate_commands(fasta_file, self.features["long"])

        if not long:
            command_2 = self.generate_commands(fasta_file, self.features["possum"]["short"])
            command.extend(command_2)
        
        # using shlex.split to parse the strings into lists for Popen class
        run_program_subprocess(command, "Possum programs")
        

@dataclass(slots=True)
class IfeatureFeatures:
    """
    A class used to represent the features of iFeature.

    Attributes
    ----------
    program : str | Path
        The program used for iFeature. Default is "iFeature".
    output : str | Path
        The output directory where the results will be stored. Default is "ifeature_features".
    drop_file : str | Path | None
        The file to be dropped. Default is None.
    drop : Iterable[str]
        An iterable containing the features to be dropped. Default is an empty tuple.
    """
    program: str | Path = "iFeature"
    output: str | Path = "ifeature_features"
    drop_file: str | Path | None = None
    drop: Iterable[str] = ()
    features: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.program = f"{self.program}/iFeature.py"
        self.features = return_features("ifeature", self.drop_file, self.drop)

        for key, feat in self.features.items():
            self.features[key] = list(set(feat).difference(self.drop))
        
        Path(self.output).mkdir(parents=True, exist_ok=True)

        print(f"iFeature features to be extracted: {self.features}")

    def generate_commands(self, fasta_file: str | Path, programs: list[str]) -> list[str]:
        """
        Extraction of features for ifeature features

        Parameters
        ----------
        fasta_file: str
            path to the different fasta files
        programs: list
            list of the different programs to run
        
        Returns
        -------
        list[str]
            list of the commands to run
        """
        num = Path(fasta_file).stem.split("_")[1]
        command = [f"python3 {self.program} --file {fasta_file} --type {prog} --out {self.output}/{prog}_{num}.tsv" for
                   prog in programs]

        return command
    
    def extract(self, fasta_file: str | Path, long: bool=False):
        """
        run the ifeature programme using subprocess

        Parameters
        ----------
        fasta_file: str
            path to the different fasta_files
        long: bool, optional
            To only extract long features or not, by default False
        """
        # ifeature features
        command = self.generate_commands(fasta_file, self.features["ifeature"]["long"])
        if not long:
            command_2 = self.generate_commands(fasta_file, self.features["ifeature"]["short"])
            command.extend(command_2)

        run_program_subprocess(command, "Ifeature programs")


def return_features(program: str, drop_file: str | Path |None=None, 
                    drop: Iterable[str]=()) -> dict:
    """
    A function to return the features to be extracted

    Parameters
    ----------
    program : str
        possum of ifeature features
    drop_file : str | Path | None, optional
        file with the features to skip, by default None
    drop : Iterable[str], optional
        An array of features to skip, by default ()

    Returns
    -------
    dict
        A dictionary of the features to be extracted filtered by drop
    """
    features = {"possum": {"long": ["pssm_cc", "tri_gram_pssm"], 
                           "short": ["aac_pssm", "ab_pssm", "d_fpssm", "dp_pssm", 
                                     "dpc_pssm", "edp", "eedp", "rpm_pssm",
                                   "k_separated_bigrams_pssm", "pssm_ac", "pssm_composition", 
                                   "rpssm", "s_fpssm", "tpc", "smoothed_pssm:5", 
                                   "smoothed_pssm:7", "smoothed_pssm:9", "pse_pssm:1", 
                                   "pse_pssm:2", "pse_pssm:3"]},

                "ifeature": {"long": ["Moran", "Geary", "NMBroto"],
                             "short": ["APAAC", "PAAC", "CKSAAGP", "CTDC", "CTDT", 
                                       "CTDD", "CTriad", "GDPC", "GTPC", "QSOrder",
                                       "SOCNumber", "GAAC", "KSCTriad"]}}

    if drop_file:
        with open(drop_file) as file:
            drop = [x.strip() for x in file.readlines()]
    if isinstance(drop, str):
        drop = (drop, )

    filtered_features = {}
    for key, value in features[program].items():
        filtered_features[key] = list(set(value).difference(drop))
    
    return filtered_features

    
def read_ifeature(features: dict[str, list[str]], length: int, 
                  ifeature_out: str|Path="ifeature_features") -> pd.DataFrame:
    """
    A function to read features from ifeatures

    Parameters
    ___________
    features: array
        An array of the features files to be read
    lenght: int
        The number of splits the input fasta has (ther number of group_* files)
    ifeature_out: str, Path
        The directory where the ifeature features are
    """
    # ifeature features
    feat = {}
    extract = features["long"]
    extract.extend(features["short"])
    for x in extract:
        feat[x] = [pd.read_csv(f"{ifeature_out}/{x}_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
    # concat features if length > 1 else return the dataframe
    for x, v in feat.items():
        val: pd.DataFrame = pd.concat(v)
        val.columns = [f"{c}_{x}" for c in val.columns]
        feat[x] = val

    all_data = pd.concat(feat.values(), axis=1)
    # change the column names

    return all_data

def read_possum(features: dict[str, list[str]], length: int, index: Iterable[str|int] | None=None, 
                possum_out: str|Path="possum_features") -> pd.DataFrame:
    """
    This function will read possum features

    Parameters
    ___________
    features: array
        An array of the features files to be read
    index: array
        An array of the indices for the possum features
    lenght: int
        The number of splits the input fasta has (ther number of group_* files)
    possum_out: str, Path
        The directory for the possum extractions
    """
    feat = {}
    extract = features["long"]
    extract.extend(features["short"])
    for x in extract:
        feat[x] = [pd.read_csv(f"{possum_out}/{x}_{i+1}.csv") for i in range(length)]
        if ":" in x:
            name = x.split(':')
            feat[x] = [pd.read_csv(f"{possum_out}/{name[0]}_{name[1]}_{i+1}.csv") for i in range(length)]

    # concat if length > 1 else return the dataframe
    for key, value in feat.items():
        val: pd.DataFrame = pd.concat(value)
        if index:
            val.index = index
        else:
            val.reset_index(drop=True, inplace=True)
        if ":" in key:
            name = key.split(":")
            val.columns = [f"{x}_{name[1]}" for x in val.columns]
        feat[key] = val

    everything = pd.concat(feat.values(), axis=1)

    return everything


def filter_features(new_features: pd.DataFrame, training_features: pd.DataFrame, 
                    output_features: str|Path="new_features.csv") -> None:
    """
    Filter the obtained features for the new samples based on the selected_features from the training samples

    Parameters
    __________
    new_features: pd.DataFrame
        The features from the new samples
    training_features: pd.DataFrame
        The features from the training samples
    output_features: str
        The path to the output file
    
    """
    extracted_out = Path(output_features)
    extracted_out.parent.mkdir(parents=True, exist_ok=True)
    feat =  new_features[training_features.columns]
    feat.to_csv(extracted_out)


def main():
    fasta_file, pssm_dir, ifeature_dir, possum_dir, ifeature_out, possum_out, extracted_out, purpose, \
    long, run, num_thread, drop, drop_file, sheets, training_features, new_features = arg_parse()

    if "extract" in purpose:
        # feature extraction
        extract = ExtractFeatures(fasta_file)
        extract.separate_bunch()
        file = list(extract.fasta_file.parent.glob("group_*.fasta"))
        possum = PossumFeatures(pssm_dir, possum_out, possum_dir, drop_file, drop)
        ifeature = IfeatureFeatures(ifeature_dir, ifeature_out, drop_file, drop)
        possum_extract = partial(possum.extract, long=long)
        ifeature_extract = partial(ifeature.extract, long=long)
        if "possum" in run:
            extract.run_extraction_parallel(file, num_thread, possum_extract)
        if "ifeature" in run:  
            extract.run_extraction_parallel(file, num_thread, ifeature_extract)

    if "read" in purpose:
        file = list(Path(fasta_file).parent.glob("group_*.fasta"))
        features = []
        if "ifeature" in run:
            ifeature_dict = return_features("ifeature", drop_file, drop)
            ifeature_features = read_ifeature(ifeature_dict, length=len(file), ifeature_out=ifeature_out)
            features.append(ifeature_features)
        if "possum" in run:
            possum_dict = return_features("possum", drop_file, drop)
            possum_features = read_possum(possum_dict, length=len(file), possum_out=possum_out)
            features.append(possum_features)
        if len(features) == 2 :
            possum_features.index == ifeature_features.index
            features = [ifeature_features, possum_features]
        every_features = pd.concat(features, axis=1)
        every_features.to_csv(f"{extracted_out}/every_features.csv")

    if "filter" in purpose:
        # feature filtering
        if not sheets:
            raise ValueError("you have not defined the selected feature sets")
        train_features = DataParser(training_features, sheets=sheets)
        if new_features:
            every_features = new_features
        try:
            new_features = training_features.read_features(every_features)
        except NameError as exe:
            raise NameError("You have not defined the new features from the new samples to perform the filtering, use -t argument") from exe
        filter_features(new_features, train_features.features, Path(extracted_out) / "new_features.csv")


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()