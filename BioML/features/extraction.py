import argparse
import shutil
from Bio import SeqIO
from Bio.SeqIO import FastaIO
import shlex
from subprocess import Popen, PIPE
import time
import pandas as pd
from ..utilities import rewrite_possum
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable
from functools import partial
from ..training.base import DataParser


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
    parser.add_argument("-r", "--run", required=False, choices=("possum", "ifeature", "both"), default="both",
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

    def __init__(self, fasta_file: str | Path, pssm_dir: str ="pssm",
                 ifeature_out: str | Path="ifeature_features", possum_out: str | Path="possum_features",
                 ifeature_dir: str ="iFeature", possum_dir: str ="POSSUM_Toolkit", 
                 drop: str | Iterable[str] =(), drop_file: str | Path | None=None):
        """
        Initialize the ExtractFeatures class

        Parameters
        ___________
        fasta_file: str
            The fasta file to be analysed
        pssm_dir: str, optional
            The directory of the generated pssm files
        ifeature: str, optional
            A path to the iFeature programme
        ifeature_out: str, optional
            A directory for the extraction results from iFeature
        possum: str, optional
            A path to the POSSUM programme
        possum_out: str, optional
            A directory for the extraction results from possum
        """
        self.fasta_file = Path(fasta_file)
        self.pssm_dir = pssm_dir
        self.ifeature = f"{ifeature_dir}/iFeature.py"
        self.possum = f"{possum_dir}"
        rewrite_possum(self.possum)
        self.ifeature_out = Path(ifeature_out)
        self.possum_out = Path(possum_out)
        self.features = {"possum": {"long": ["pssm_cc", "tri_gram_pssm"], 
                               "short": ["aac_pssm", "ab_pssm", "d_fpssm", "dp_pssm", "dpc_pssm", "edp", "eedp", "rpm_pssm",
                                "k_separated_bigrams_pssm", "pssm_ac", "pssm_composition", "rpssm", "s_fpssm", "tpc"],
                                "special": ["smoothed_pssm:5", "smoothed_pssm:7", "smoothed_pssm:9", 
                                            "pse_pssm:1", "pse_pssm:2", "pse_pssm:3"]},
                        "ifeature": {"long": ["Moran", "Geary", "NMBroto"], "short": 
                                 ["APAAC", "PAAC", "CKSAAGP", "CTDC", "CTDT", "CTDD", "CTriad", "GDPC", "GTPC", "QSOrder",
                                  "SOCNumber", "GAAC", "KSCTriad"]}}

        if drop_file:
            with open(drop_file) as file:
                drop = [x.strip() for x in file.readlines()]
        if type(drop) == str:
            drop = (drop, )

        for key, value in self.features.items():
            for k, v in value.items():
                self.features[key][k] = list(set(v).difference(drop))

        print(f"Extracting iFeature features {self.features['ifeature']}")
        print(f"Extracting Possum features {self.features['possum']}")
 

    @staticmethod
    def _batch_iterable(iterable, batch_size):
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

    def _separate_bunch(self):
        """
        A class that separates the fasta files into smaller fasta files

        """
        with open(self.fasta_file) as inp:
            record = list(SeqIO.parse(inp, "fasta"))
            if len(record) > 5_000:
                for i, batch in enumerate(self._batch_iterable(record, 5_000)):
                    filename = f"group_{i+1}.fasta"
                    with open(f"{self.fasta_file.parent}/{filename}", "w") as split:
                        print(f"{self.fasta_file.parent}/{filename}")
                        fasta_out = FastaIO.FastaWriter(split, wrap=None)
                        fasta_out.write_file(batch)
                del record
            else:
                shutil.copyfile(self.fasta_file, f"{self.fasta_file.parent}/group_1.fasta")

    @staticmethod
    def run_progam(commands: str, program_name: str | None=None):
        """
        Run in parallel the subprocesses from the command
        Parameters
        ----------
        commands: list[str]
            A list of commandline commands that calls to Possum programs or ifeature programs
        program_name: str, optional
            A name to identify the commands
        """
        proc = [Popen(shlex.split(command), stderr=PIPE, stdout=PIPE, text=True) for command in commands]
        start = time.time()
        for p in proc:
            output, errors = p.communicate()
            with open(f"error_file.txt", "a") as out:
                out.write(f"{output}")
                out.write(f"{errors}")
        end = time.time()
        if program_name:
            print(f"start running {program_name}")
        print(f"It took {end - start} second to run")

    def ifeature_commands(self, fasta_file: str | Path, programs: list[str]):
        """
        Extraction of features for ifeature features

        Parameters
        ----------
        fasta_file: str
            path to the different fasta files
        """
        num = Path(fasta_file).stem.split("_")[1]
        commands_1 = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in programs]

        return commands_1
    
    def possum_special_commands(self, fasta_file: str | Path, programs: list[str]):

        num = Path(fasta_file).stem.split("_")[1]
        command_2_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog.split(":")[0]} -a {prog.split(":")[1]} -o '
            f'{self.possum_out}/{prog.split(":")[0]}_{prog.split(":")[1]}_{num}.csv' for prog in
                programs]
        return command_2_possum

    def possum_commands(self, fasta_file: str | Path, programs: list[str]):
        """
        Writing the commands to run the possum features that take a lot of time

        Parameters
        ----------
        fasta_file: str
            path to the different files
        """
        num = Path(fasta_file).stem.split("_")[1]
        command_3_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in programs]

        return command_3_possum

    def extraction_ifeature(self, fasta_file: str | Path, long: bool=False):
        """
        run the ifeature programme iteratively

        Parameters
        ----------
        fasta_file: str
            path to the different fasta_files
        """
        # ifeature features
        commands_1 = self.ifeature_commands(fasta_file, self.features["ifeature"]["long"])
        if not long:
            commands_2 = self.ifeature_commands(fasta_file, self.features["ifeature"]["short"])
            commands_1.extend(commands_2)

        self.run_progam(commands_1, "Ifeature programs")

    def extraction_possum(self, fasta_file: str | Path, long: bool=False):
        """
        run the possum programme iteratively

        Parameters
        ----------
        fasta_file: str
            Path to the different fasta files
        """
        # possum features
        command_1_possum = self.possum_commands(fasta_file, self.features["possum"]["long"])

        if not long:
            command_2_possum = self.possum_commands(fasta_file, self.features["possum"]["short"])
            command_3_possum = self.possum_special_commands(fasta_file, self.features["possum"]["special"])
            command_1_possum.extend(command_2_possum)
            command_1_possum.extend(command_3_possum)
        
        # using shlex.split to parse the strings into lists for Popen class
        self.run_progam(command_1_possum, "Possum programs")

    def run_extraction_parallel(self, num_thread: int, run: str ="both", long: bool=False):
        """
        Using a pool of workers to run the 2 programmes

        Parameters
        ----------
        long:
            If to run only the longer features
        """
        self.possum_out.mkdir(parents=True, exist_ok=True)
        self.ifeature_out.mkdir(parents=True, exist_ok=True)
        file = list(self.fasta_file.parent.glob(f"group_*.fasta"))
        if not file:
            self._separate_bunch()
        
        file.sort(key=lambda x: int(x.stem.split("_")[1]))
        with get_context("spawn").Pool(processes=num_thread) as pool:
            if run == "both":
                pool.map(partial(self.extraction_ifeature, long=long), file)
                pool.map(partial(self.extraction_possum, long=long), file)
            elif run == "possum":
                pool.map(partial(self.extraction_possum, long=long), file)
            elif run == "ifeature":
                pool.map(partial(self.extraction_ifeature, long=long), file)


class ReadFeatures:
    """
    A class to read the generated features
    """
    def __init__(self, group_file_path: str, ifeature_out: str="ifeature_features", possum_out: str="possum_features",
                 drop: Iterable[str]=(), drop_file: str | Path | None=None):
        """
        Initialize the class ReadFeatures

        Parameters
        ___________
        group_file_path: str
            The path to where the splited files from the original fasta file are. They are name group_*.fasta
        ifeature_out: str, optional
            A directory for the extraction results from iFeature
        possum_out: str, optional
            A directory for the extraction results from possum
        extracted_out: str, optional
            A directory to store the filtered features from all the generated features
        drop: list, optional
            A list of the features to drop
        drop_file: str, optional
            The path to the file with the features to drop
        
        """
        self.ifeature_out = ifeature_out
        self.possum_out = possum_out
        self.grup_file_path = Path(group_file_path)

        self.features = {"possum": {"normal": ["pssm_cc", "tri_gram_pssm", "aac_pssm", "ab_pssm", "d_fpssm", "dp_pssm", 
                                               "dpc_pssm", "edp", "eedp", "rpm_pssm", "k_separated_bigrams_pssm", 
                                               "pssm_ac", "pssm_composition", "rpssm", "s_fpssm", "tpc"],

                                    "special": ["smoothed_pssm:5", "smoothed_pssm:7", "smoothed_pssm:9", 
                                            "pse_pssm:1", "pse_pssm:2", "pse_pssm:3"]},

                        "ifeature": {"all" : ["Moran", "Geary", "NMBroto", "APAAC", "PAAC", "CKSAAGP", "CTDC", "CTDT", "CTDD", 
                                     "CTriad", "GDPC", "GTPC", "QSOrder", "SOCNumber", "GAAC", "KSCTriad"]}}

        if drop_file:
            with open(drop_file) as file:
                drop = [x.strip() for x in file.readlines()]
        if type(drop) == str:
            drop = (drop, )

        for key, value in self.features.items():
            for k, v in value.items():
                self.features[key][k] = list(set(v).difference(drop))

        print(f"Reading iFeature features {self.features['ifeature']}")
        print(f"Reading Possum features {self.features['possum']}")

    def read_ifeature(self, length: int):
        """
        A function to read features from ifeatures

        Parameters
        ___________
        lenght: int
            The number of splits the input fasta has (ther number of group_* files)
        """
        # ifeature features
        feat = {}
        for x in self.features["ifeature"]["all"]:
            feat[x] = [pd.read_csv(f"{self.ifeature_out}/{x}_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        # concat features if length > 1 else return the dataframe
        for x, v in feat.items():
            val = pd.concat(v)
            val.columns = [f"{c}_{x}" for c in val.columns]
            feat[x] = val

        all_data = pd.concat(feat.values(), axis=1)
        # change the column names

        return all_data

    def read_possum(self, ID: Iterable[str|int], length: int):
        """
        This function will read possum features

        Parameters
        ___________
        ID: array
            An array of the indices for the possum features
        lenght: int
            The number of splits the input fasta has (ther number of group_* files)
        """
        feat = {}
        for x in self.features["possum"]["normal"]:
            feat[x] = [pd.read_csv(f"{self.possum_out}/{x}_{i+1}.csv") for i in range(length)]
        # reads features of possum
        for x in self.features["possum"]["special"]:
            name = x.split(':')
            feat[x] = [pd.read_csv(f"{self.possum_out}/{name[0]}_{name[1]}_{i+1}.csv") for i in range(length)]

        # concat if length > 1 else return the dataframe
        for key, value in feat.items():
            val = pd.concat(value)
            val.index = ID
            if "smoothed" in key or "pse_pssm" in key:
                name = key.split(":")
                val.columns = [f"{x}_{name[1]}" for x in val.columns]
            feat[key] = val

        everything = pd.concat(feat.values(), axis=1)

        return everything

    def read(self):
        """
        Reads all the features
        """
        file = list(self.group_file_path.glob("group_*.fasta"))
        all_data = self.read_ifeature(len(file))
        everything = self.read_possum(all_data.index, len(file))
        # concatenate the features
        features = pd.concat([all_data, everything], axis=1)
        return features
    

def filter_features(new_features: pd.DataFrame, training_features: pd.DataFrame, output_features="new_features.csv") -> None:
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
        extract = ExtractFeatures(fasta_file, pssm_dir, ifeature_out, possum_out, ifeature_dir, possum_dir, drop, drop_file)
        extract.run_extraction_parallel(num_thread, run, long)

    if "read" in purpose:
        filtering = ReadFeatures(Path(fasta_file).parent, ifeature_out, possum_out, drop, drop_file)
        every_features = filtering.read()
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
        except NameError:
            raise NameError("You have not defined the new features from the new samples to perform the filtering, use -t argument")
        filter_features(new_features, train_features.features, Path(extracted_out) / "new_features.csv")


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()