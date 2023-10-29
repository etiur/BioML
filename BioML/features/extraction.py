import argparse
import shutil
from Bio import SeqIO
from Bio.SeqIO import FastaIO
import shlex
from subprocess import Popen, PIPE
import time
import pandas as pd
from BioML.utilities import write_excel, rewrite_possum
from os.path import basename
from multiprocessing import get_context
from pathlib import Path


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
    parser.add_argument("-e", "--excel", required=False,
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
    parser.add_argument("-t", "--type", required=False, default="all", nargs="+", choices=("all", "APAAC", "PAAC",
                        "CKSAAGP", "Moran", "Geary", "NMBroto", "CTDC", "CTDT", "CTDD", "CTriad", "GDPC", "GTPC",
                        "QSOrder", "SOCNumber", "GAAC", "KSCTriad", "aac_pssm", "ab_pssm", "d_fpssm", "dp_pssm",
                        "dpc_pssm", "edp", "eedp", "rpm_pssm", "k_separated_bigrams_pssm", "pssm_ac", "pssm_cc",
                        "pssm_composition", "rpssm", "s_fpssm", "smoothed_pssm:5", "smoothed_pssm:7", "smoothed_pssm:9",
                        "tpc", "tri_gram_pssm", "pse_pssm:1", "pse_pssm:2", "pse_pssm:3"),
                        help="A list of the features to extract")
    parser.add_argument("-tf", "--type_file", required=False, help="the path to the file with the feature names, separated by spaces or newlines")
    parser.add_argument("-s", "--sheets", required=False, nargs="+",
                        help="Names of the selected sheets from the features and the "
                             "index of the models in this format-> sheet name:index model1,index model2 "
                             "without the spaces. If only name of the sheets, it is assumed that all kfold "
                             "models are selected. It is possible to have one sheet with kfold indices but in another "
                             "ones without")

    args = parser.parse_args()

    return [args.fasta_file, args.pssm_dir, args.ifeature_dir, args.possum_dir, args.ifeature_out,
            args.possum_out, args.extracted_out, args.purpose, args.long, args.run, args.num_thread, args.type,
            args.type_file, args.sheets, args.excel]


class ExtractFeatures:
    """
    A class to extract features using Possum and iFeatures
    """

    def __init__(self, fasta_file, pssm_dir="pssm",
                 ifeature_out="ifeature_features", possum_out="possum_features",
                 ifeature_dir="iFeature",
                 thread=12, run="both", possum_dir="POSSUM_Toolkit", types="all",
                 type_file=None):
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
        if (self.fasta_file.parent / "no_short.fasta").exists():
            self.fasta_file = self.fasta_file.parent / "no_short.fasta"
        self.pssm_dir = pssm_dir
        self.ifeature = f"{ifeature_dir}/iFeature.py"
        self.possum = f"{possum_dir}/possum_standalone.pl"
        rewrite_possum(self.possum)
        self.ifeature_out = Path(ifeature_out)
        self.possum_out = Path(possum_out)
        self.thread = thread
        self.run = run
        self.pos_short = ["aac_pssm", "ab_pssm", "d_fpssm", "dp_pssm", "dpc_pssm", "edp", "eedp", "rpm_pssm",
                          "k_separated_bigrams_pssm", "pssm_ac", "pssm_composition", "rpssm", "s_fpssm", "tpc"]
        self.pse_pssm = ["pse_pssm:1", "pse_pssm:2", "pse_pssm:3"]
        self.smoothed_pssm = ["smoothed_pssm:5", "smoothed_pssm:7", "smoothed_pssm:9"]
        self.ifea_short = ["APAAC", "PAAC", "CKSAAGP", "CTDC", "CTDT", "CTDD", "CTriad", "GDPC", "GTPC", "QSOrder",
                           "SOCNumber", "GAAC", "KSCTriad"]
        self.pos_long = ["pssm_cc", "tri_gram_pssm"]
        self. ifea_long = ["Moran", "Geary", "NMBroto"]
        if type_file:
            with open(type_file) as file:
                types = [x.strip() for x in file.readlines()]
        if "all" not in types:
            for f in self.pse_pssm:
                if f not in types:
                    self.pse_pssm.remove(f)
            for f in self.smoothed_pssm:
                if f not in types:
                    self.smoothed_pssm.remove(f)
            for f in self.pos_long:
                if f not in types:
                    self.pos_long.remove(f)
            for f in self.pos_short:
                if f not in types:
                    self.pos_short.remove(f)
            for f in self.ifea_short:
                if f not in types:
                    self.ifea_short.remove(f)
            for f in self.ifea_long:
                if f not in types:
                    self.ifea_long.remove(f)
            print(f"Extracting iFeature features {self.ifea_long + self.ifea_short}")
            print(f"Extracting Possum features {self.pos_long + self.pos_short + self.smoothed_pssm, self.pse_pssm}")
 
        else:
            print("Extracting all features should only be used for training new models")

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
    def run_progam(commands, program_name=None):
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

    def ifeature_short(self, fasta_file):
        """
        Extraction of features that are fast from ifeature

        Parameters
        ----------
        fasta_file: str
            path to the different fasta files
        """
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]

        commands_1 = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in self.ifea_short]
        return commands_1

    def ifeature_long(self, fasta_file):
        """
        Extraction of features for time-consuming ifeature features

        Parameters
        ----------
        fasta_file: str
            path to the different fasta files
        """
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]
        commands_1 = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in self.ifea_long]

        return commands_1

    def possum_short(self, fasta_file):
        """
        writing the commands to run the possum features that do not take a lot of time

        Parameters
        ----------
        fasta_file: str
            path to the different files
        """
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]

        command_1_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in self.pos_short]
        if self.pse_pssm or self.smoothed_pssm:
            command_2_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog.split(":")[0]} -a {prog.split(":")[1]} -o '
            f'{self.possum_out}/{prog.split(":")[0]}_{prog.split(":")[1]}_{num}.csv' for prog in
                self.pse_pssm + self.smoothed_pssm]

        else:
            command_2_possum = []

        command_1_possum.extend(command_2_possum)

        return command_1_possum

    def possum_long(self, fasta_file):
        """
        Writing the commands to run the possum features that take a lot of time

        Parameters
        ----------
        fasta_file: str
            path to the different files
        """
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]
        command_3_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in self.pos_long]

        return command_3_possum

    def extraction_long(self, fasta_file):
        """
        Writing the commands to run the features that take a lot of time

        Parameters
        ----------
        fasta_file: str
            path to the different files
        """
        # generate the commands
        commands_1 = self.ifeature_long(fasta_file)
        command_3_possum = self.possum_long(fasta_file)
        commands_1.extend(command_3_possum)
        self.run_progam(commands_1, "All long features")

    def extraction_all(self, fasta_file):
        """
        Writing the commands to run the all programmes

        Parameters
        ----------
        fasta_file: str
            Path to the different fasta files
        """
        # ifeature features
        commands_1 = self.ifeature_short(fasta_file)
        commands_1_long = self.ifeature_long(fasta_file)
        # possum features
        command_1_possum = self.possum_short(fasta_file)
        command_3_possum = self.possum_long(fasta_file)
        # combine the commands
        commands_1.extend(command_1_possum)
        commands_1.extend(commands_1_long)
        commands_1.extend(command_3_possum)
        self.run_progam(commands_1, "All features")

    def extraction_ifeature(self, fasta_file):
        """
        run the ifeature programme iteratively

        Parameters
        ----------
        fasta_file: str
            path to the different fasta_files
        """
        # ifeature features
        commands_1 = self.ifeature_short(fasta_file)
        commands_1_long = self.ifeature_long(fasta_file)
        # combining the commands
        commands_1.extend(commands_1_long)
        self.run_progam(commands_1, "All Ifeature programs")

    def extraction_possum(self, fasta_file):
        """
        run the possum programme iteratively

        Parameters
        ----------
        fasta_file: str
            Path to the different fasta files
        """
        # possum features
        command_1_possum = self.possum_short(fasta_file)
        command_3_possum = self.possum_long(fasta_file)
        # combining all the commands
        command_1_possum.extend(command_3_possum)
        # using shlex.split to parse the strings into lists for Popen class
        self.run_progam(command_1_possum, "All Possum programs")

    def run_extraction_parallel(self, long=None):
        """
        Using a pool of workers to run the 2 programmes

        Parameters
        ----------
        long:
            If to run only the longer features
        """
        self.possum_out.mkdir(parents=True, exist_ok=True)
        self.ifeature_out.mkdir(parents=True, exist_ok=True)
        name = self.fasta_file.parent/"group_1.fasta"
        if not name.exists():
            self._separate_bunch()
        file = list(self.fasta_file.parent.glob(f"group_*.fasta"))
        file.sort(key=lambda x: int(basename(x).replace(".fasta", "").split("_")[1]))
        with get_context("spawn").Pool(processes=self.thread) as pool:
            if self.run == "both":
                if not long:
                    pool.map(self.extraction_all, file)
                else:
                    pool.map(self.extraction_long, file)
            elif self.run == "possum":
                if not long:
                    pool.map(self.extraction_possum, file)
                else:
                    pool.map(self.possum_long, file)
            elif self.run == "ifeature":
                if not long:
                    pool.map(self.extraction_ifeature, file)
                else:
                    pool.map(self.ifeature_long, file)


class ReadFeatures:
    """
    A class to read the generated features
    """
    def __init__(self, fasta_file, ifeature_out="ifeature_features", possum_out="possum_features",
                 extracted_out="extracted_features", types="all", type_file=None, excel_feature_file=None, sheets=()):
        """
        Initialize the class ReadFeatures

        Parameters
        ___________
        fasta_file: str
            The name of the fasta file
        ifeature_out: str, optional
            A directory for the extraction results from iFeature
        possum_out: str, optional
            A directory for the extraction results from possum
        filtered_out: str, optional
            A directory to store the filtered features from all the generated features
        types: list[str]
            A list of features to be read
        
        """
        self.ifeature_out = ifeature_out
        self.possum_out = possum_out
        self.features = None
        self.excel_feature = excel_feature_file
        self.extracted_out = Path(extracted_out)
        self.extracted_out.mkdir(parents=True, exist_ok=True)
        self.base = Path(fasta_file).parent

        self.poss = ["aac_pssm", "ab_pssm", "d_fpssm", "dp_pssm", "dpc_pssm", "edp", "eedp", "rpm_pssm",
                     "k_separated_bigrams_pssm", "pssm_ac", "pssm_composition", "rpssm", "s_fpssm", "tpc",
                     "pssm_cc", "tri_gram_pssm"]
        self.pse_pssm = ["pse_pssm:1", "pse_pssm:2", "pse_pssm:3"]
        self.smoothed_pssm = ["smoothed_pssm:5", "smoothed_pssm:7", "smoothed_pssm:9"]

        self. ifea = ["APAAC", "PAAC", "CKSAAGP", "CTDC", "CTDT", "CTDD", "CTriad", "GDPC", "GTPC", "QSOrder",
                      "SOCNumber", "GAAC", "KSCTriad", "Moran", "Geary", "NMBroto"]
        if type_file:
            with open(type_file) as file:
                types = [x.strip() for x in file.readlines()]
        if "all" not in types:
            for f in self.pse_pssm:
                if f not in types:
                    self.pse_pssm.remove(f)
            for f in self.smoothed_pssm:
                if f not in types:
                    self.smoothed_pssm.remove(f)
            for f in self.poss:
                if f not in types:
                    self.poss.remove(f)
            for f in self.ifea:
                if f not in types:
                    self.ifea.remove(f)
            print(f"Reading iFeature features {self.ifea}")
            print(f"Reading Possum features {self.poss + self.pse_pssm + self.smoothed_pssm}")
        else:
            print("Reading all features used only for training")
        if sheets:
            self.selected_sheets = []
            self.selected_kfolds = {}
            self._check_sheets(sheets)

    def _check_sheets(self, sheets):
        for x in sheets:
            indices = x.split(":")
            sh = indices[0]
            if sh.isdigit():
                sh = int(sh)
            self.selected_sheets.append(sh)
            if len(indices) > 1:
                kfold = tuple(int(x) for x in indices[1].split(","))
                self.selected_kfolds[sh] = kfold
            else:
                self.selected_kfolds[sh] = ()

    def read_ifeature(self, length):
        """
        A function to read features from ifeatures
        lenght: int
            The number of splits the input fasta has
        """
        # ifeature features
        feat = {}
        for x in self.ifea:
            feat[x] = [pd.read_csv(f"{self.ifeature_out}/{x}_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        # concat features if length > 1 else return the dataframe
        if length > 1:
            for x, v in feat.items():
                val = pd.concat(v)
                val.columns = [f"{c}_{x}" for c in val.columns]
                feat[x] = val
        else:
            for x, v in feat.items():
                val = v[0]
                val.columns = [f"{c}_{x}" for c in val.columns]
                feat[x] = val

        all_data = pd.concat(feat.values(), axis=1)
        # change the column names

        return all_data

    def read_possum(self, ID, length):
        """
        This function will read possum features

        Parameters
        ___________
        ID: array
            An array of the indices for the possum features
        """
        feat = {}
        for x in self.poss:
            feat[x] = [pd.read_csv(f"{self.possum_out}/{x}_{i+1}.csv") for i in range(length)]
        # reads features of possum
        for x in self.pse_pssm + self.smoothed_pssm:
            name = x.split(':')
            feat[x] = [pd.read_csv(f"{self.possum_out}/{name[0]}_{name[1]}_{i+1}.csv") for i in range(length)]

        # concat if length > 1 else return the dataframe
        if length > 1:
            for key, value in feat.items():
                val = pd.concat(value)
                val.index = ID
                if "smoothed" in key or "pse_pssm" in key:
                    name = key.split(":")
                    val.columns = [f"{x}_{name[1]}" for x in val.columns]
                feat[key] = val
        else:
            for key, value in feat.items():
                val = value[0]
                val.index = ID
                if "smoothed" in key or "pse_pssm" in key:
                    name = key.split(":")
                    val.columns = [f"{x}_{name[1]}" for x in val.columns]
                feat[key] = val
        # Possum features

        everything = pd.concat(feat.values(), axis=1)

        return everything

    def read(self):
        """
        Reads all the features
        """
        file = list(self.base.glob("group_*.fasta"))
        all_data = self.read_ifeature(len(file))
        everything = self.read_possum(all_data.index, len(file))
        # concatenate the features
        self.features = pd.concat([all_data, everything], axis=1)
        return self.features

    def filter_features(self):
        """
        filter the obtained features based on the reference_feature_file (self.excel_feature)
        """
        self.read()
        # The reference excel has a multicomlun were the first level are the kfolds and the second level are the features selected for that kfold
        training_features = pd.read_excel(self.excel_feature, index_col=0, sheet_name=self.selected_sheets, header=[0, 1],
                                          engine='openpyxl')
        with pd.ExcelWriter(self.extracted_out/"new_features.xlsx", mode="w", engine="openpyxl") as writer:
            for sheet, feature in training_features.items():
                feature_dict = {}
                for col in feature.columns.unique(level=0):
                    ind = int(col.split("_")[-1])
                    if self.selected_kfolds[sheet] and ind not in self.selected_kfolds[sheet]: continue
                    sub_feat = feature.loc[:, f"split_{ind}"]
                    feature_dict[f"split_{ind}"] = self.features[sub_feat.columns]
                write_excel(writer, pd.concat(feature_dict, axis=1), f"{sheet}")


def extract_and_filter(fasta_file=None, pssm_dir="pssm", ifeature_out="ifeature_features",
                       possum_dir="/gpfs/home/bsc72/bsc72661/feature_extraction/POSSUM_Toolkit",
                       ifeature_dir="/gpfs/projects/bsc72/ruite/enzyminer/iFeature", possum_out="possum_features",
                       extracted_out="training_features", purpose=("extract", "read"), long=False, thread=100,
                       run="both", types="all", type_file=None, excel_feature_file=None, selected=()):
    """
    A function to extract and filter the features

    Parameters
    __________
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
    filtered_out: str, optional
        A directory to store the filtered features from all the generated features
    thread: int
        The number of pool workers to use to run the programmes
    run: str
        which programme to run
    """
    # Feature extraction for both training or prediction later
    if "extract" in purpose:
        extract = ExtractFeatures(fasta_file, pssm_dir, ifeature_out, possum_out, ifeature_dir, thread, run,
                                  possum_dir, types, type_file)
        extract.run_extraction_parallel(long)
    # feature reading for the training
    if "read" in purpose:
        filtering = ReadFeatures(fasta_file, ifeature_out, possum_out, extracted_out, types, type_file,
                                 excel_feature_file, selected)
        every_features = filtering.read()
        every_features.to_csv(f"{extracted_out}/every_features.csv")
    # feature extraction for prediction
    if "filter" in purpose:
        # feature filtering
        if not selected:
            raise ("you have not defined the selected feature sets")
        filtering = ReadFeatures(fasta_file, ifeature_out, possum_out, extracted_out, types, type_file,
                                 excel_feature_file, selected)
        filtering.filter_features()


def main():
    fasta_file, pssm_dir, ifeature_dir, possum_dir, ifeature_out, possum_out, extracted_out, purpose, \
    long, run, num_thread, types, type_file, sheets, excel = arg_parse()

    extract_and_filter(fasta_file, pssm_dir, ifeature_out, possum_dir, ifeature_dir, possum_out,
                       extracted_out, purpose, long, num_thread, run, types, type_file, excel, sheets)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()