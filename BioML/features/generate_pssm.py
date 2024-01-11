from Bio.Blast.Applications import NcbimakeblastdbCommandline as makedb
from Bio.Blast.Applications import NcbipsiblastCommandline as psiblast
import argparse
import os
from pathlib import Path
import time
from multiprocessing import get_context
import shutil
from Bio import SeqIO
from Bio.SeqIO import FastaIO
from subprocess import call
import shlex
from ..utilities.utils import rewrite_possum, MmseqsClustering


def arg_parse():
    parser = argparse.ArgumentParser(description="creates a database and performs psiblast")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path", required=False)
    parser.add_argument("-f", "--fasta_dir", required=False, help="The directory for the fasta files",
                        default="fasta_files")
    parser.add_argument("-p", "--pssm_dir", required=False, help="The directory for the pssm files",
                        default="pssm")
    parser.add_argument("-di", "--dbinp", required=False, help="The path to the fasta files to create the database")
    parser.add_argument("-do", "--dbout", required=False, help="The name for the created database",
                        default="database/uniref50")
    parser.add_argument("-n", "--num_thread", required=False, default=100, type=int,
                        help="The number of threads to use for the generation of pssm profiles")
    parser.add_argument("-num", "--number", required=False, help="a number for the files", default="*")
    parser.add_argument("-iter", "--iterations", required=False, default=3, type=int, help="The number of iterations "
                                                                                         "in PSIBlast")
    parser.add_argument("-Po", "--possum_dir", required=False, help="A path to the possum programme",
                        default="POSSUM_Toolkit/")
    parser.add_argument("-m", "--use_mmseqs", action="store_false", help="Use mmseqs to cluster the sequences")
    parser.add_argument("-e", "--evalue", required=False, default=0.01, type=float, help="The evalue for the mmseqs")
    parser.add_argument("-s", "--sensitivity", required=False, default=6.5, type=float, help="The sensitivity for the mmseqs")
    args = parser.parse_args()

    return [args.fasta_dir, args.pssm_dir, args.dbinp, args.dbout, args.num_thread, args.number,
            args.fasta_file, args.iterations, args.possum_dir, args.use_mmseqs, args.evalue, args.sensitivity]


class ExtractPssm:
    """
    A class to extract pssm profiles from protein sequecnes
    """
    def __init__(self, fasta: str | Path, num_threads: int=100, fasta_dir: str | Path="fasta_files", 
                 pssm_dir: str| Path="pssm", dbinp: str=None,
                 dbout: str | Path="uniref50", iterations: int=3, possum_dir: str="POSSUM_Toolkit/"):
        """
        Initialize the ExtractPssm class

        Parameters
        ___________
        fasta: str
            The file to extract the PSSM
        num_threads: int, optional
            The number of threads to use for the generation of pssm profiles
        fasta_dir: str, optional
            The directory of the fasta files
        pssm_dir: str, optional
            The directory for the output pssm files
        dbinp: str, optional
            The path to the protein fasta file to construct the database
        dbout: str, optional
            The name of the created database database
        iterations: int, optional
            The number of iterations in PSIBlast
        possum_dir: str, optional
            A path to the possum programme
        """
        self.fasta_file = Path(fasta)
        self.pssm = Path(pssm_dir)
        self.fasta_dir = Path(fasta_dir)
        self.dbinp = dbinp
        self.dbout = Path(dbout)
        self.num_thread = num_threads
        self.iter = iterations
        self.possum = f"{possum_dir}/possum_standalone.pl"
        rewrite_possum(self.possum)

    def makedata(self):
        """
        A function that creates a database for the PSI_blast
        """
        self.dbout.parent.mkdir(parents=True, exist_ok=True)
        # running the blast commands
        blast_db = makedb(dbtype="prot", input_file=f"{self.dbinp}", out=f"{self.dbout}", title=f"{self.dbout.name}")
        stdout_db, stderr_db = blast_db()

        return stdout_db, stderr_db

    def _check_pssm(self, files: str|Path):
        """
        Check if the pssm files are correct
        """
        with open(files, "r") as pssm:
            if "PSI" not in pssm.read():
                os.remove(files)

    def _check_output(self, file: str|Path):
        file = Path(file)
        if not file.exists():
            remove = Path("removed_dir")
            remove.mkdir(parents=True, exist_ok=True)
            shutil.move(self.fasta_dir/f"{file.stem}.fsa", remove/f"{file.stem}.fsa")

    def fast_check(self, num: str|int):
        """
        Accelerates the checking of files
        """
        file = list(self.pssm.glob(f"seq_{num}*.pssm"))
        with get_context("spawn").Pool(processes=self.num_thread) as executor:
            executor.map(self._check_pssm, file)

    def generate(self, file: str|Path):
        """
        A function that generates the PSSM profiles
        """
        file = Path(file)
        psi = psiblast(db=self.dbout, evalue=0.001, num_iterations=self.iter,
                       out_ascii_pssm=self.pssm/f"{file.stem}.pssm", save_pssm_after_last_round=True, query=file,
                       num_threads=self.num_thread)

        start = time.perf_counter()
        psi()
        end = time.perf_counter()
        self._check_output(self.pssm/f"{file.stem}.pssm")
        return f"it took {round((end - start)/60, 4)} min to finish {file.stem}.pssm"

    def run_generate(self, num: str|int):
        """
        run the generate function
        """
        self.fast_check(num)
        files = list(self.fasta_dir.glob(f"seq_{num}*.fsa"))
        files.sort(key=lambda x: int(x.stem.split("_")[1]))
        files = [x for x in files if not (self.pssm/f"{x.stem}.pssm").exists()]
        for file in files:
            print(f"Generate PSSM for {file}, {files.index(file)+1}/{len(files)}")
            res = self.generate(file)
            print(res)
    
    def _clean_fasta(self, length: int=100):
        """
        Clean the fasta file

        Parameters
        ==========
        length: int
            length_threshold

        """
        illegal = f"perl {self.possum}/utils/removeIllegalSequences.pl -i {self.fasta_file} -o {self.fasta_file.parent}/no_illegal.fasta"
        short = f"perl {self.possum}/utils/removeShortSequences.pl -i {self.fasta_file.parent}/no_illegal.fasta -o {self.fasta_file.parent}/no_short.fasta -n {length}"
        call(shlex.split(illegal), close_fds=False)
        call(shlex.split(short), close_fds=False)
        self.fasta_file.rename("original_fasta.fasta")
        (self.fasta_file.parent/"no_short.fasta").rename(self.fasta_file.with_stem(f"{self.fasta_file.stem}_fixed"))
        self.fasta_file = self.fasta_file.with_stem(f"{self.fasta_file.stem}_fixed")
    
    def _separate_single(self):
        """
        A function that separates the fasta files into individual files
        Returns
        
        file: iterator
            An iterator that stores the single-record fasta files
        """
        with open(self.fasta_file) as inp:
            record = SeqIO.parse(inp, "fasta")
            count = 1
            # Write the record into new fasta files
            for seq in record:
                with open(self.fasta_dir/f"seq_{count}.fsa", "w") as split:
                    fasta_out = FastaIO.FastaWriter(split, wrap=None)
                    fasta_out.write_record(seq)
                count += 1
    
    def _remove_sequences_from_input(self):
        """
        A function that removes the fasta sequences that psiblast could not generate pssm files from,
        from the input fasta file. 
        If inside the remove dir there are fasta files them you have to use this function.
        """
        # Search for fasta files that doesn't have pssm files
        fasta_files = list(Path('removed_dir').glob("seq_*.fsa"))
        difference = sorted(fasta_files, key=lambda x: int(x.stem.split("_")[1]), reverse=True)

        if len(difference) > 0 and not os.path.exists(f"{self.base}/no_short_before_pssm.fasta"):
            with open(self.fasta_file) as inp:
                record = SeqIO.parse(inp, "fasta")
                record_list = list(record)
                # Eliminate the sequences from the input fasta file and move the single fasta sequences
                # to another folder
                for files in difference:
                    num = int(files.stem.split("_")[1]) - 1
                    del record_list[num]
                    # Rename the input fasta file so to create a new input fasta file with the correct sequences
                self.fasta_file.rename(self.fasta_file.with_stem("befoe_filtering_by_pssm"))
                with open(self.fasta_file.with_stem("filtered_by_pssm"), "w") as out:
                    fasta_out = FastaIO.FastaWriter(out, wrap=None)
                    fasta_out.write_file(record_list)


def generate_pssm(fasta: str | Path, num_threads: int=100, fasta_dir: str | Path="fasta_files", pssm_dir: str | Path="pssm", 
                  dbinp: str | Path | None=None, dbout: str | Path="uniref50", num: int | str="*",
                  iterations: int=3, possum_dir: str | Path="POSSUM_Toolkit"):
    """
    A function that creates protein databases, generates the pssms and returns the list of files

    fasta: str, optional
        The file to be analysed
    num_threads: int, optional
        The number of threads to use for the generation of pssm profiles
    fasta_dir: str, optional
        The directory of the fasta files
    pssm_dir: str, optional
        The directory for the output pssm files
    dbinp: str, optional
        The path to the protein database
    dbout: str, optional
        The name of the created databse database
    num: int or *, optional
        used to glob for files: e.g. seq_num*.fsa -> where num is a integer or another * to glob for all files
    iterations: int, optional
        The number of iterations in PSIBlast
    possum_dir: str, optional
        A path to the possum programme
    """
    Path(fasta_dir).mkdir(parents=True, exist_ok=True)
    Path(pssm_dir).mkdir(parents=True, exist_ok=True)

    pssm = ExtractPssm(fasta, num_threads, fasta_dir, pssm_dir, dbinp, dbout, iterations, possum_dir)
    # generate the database if not present
    pssm._clean_fasta()
    pssm._separate_single()

    if dbinp and dbout:
        pssm.makedata()

    pssm.run_generate(num)
    pssm._remove_sequences_from_input()
    

def generate_with_mmseqs(fasta: str | Path, dbinp: str | Path | None=None, dbout: str | Path="uniref50", evalue: float =0.01, num_iterations: int=3, 
                         sensitivity: float = 6.5, num_threads: int=100,  pssm_file: str = "result.pssm", pssm_dir: str | Path="pssm"):
    generate_searchdb = False
    if dbinp is None:
        generate_searchdb = True
    MmseqsClustering.easy_generate_pssm(fasta, dbout, evalue, num_iterations, sensitivity, pssm_file, 
                                        generate_searchdb, threads=num_threads)
    MmseqsClustering.split_pssm(pssm_file, pssm_dir)


def main():
    fasta_dir, pssm_dir, dbinp, dbout, num_thread, num, fasta_file, iterations, possum_dir, \
    use_mmseqs, evalue, sensitivity = arg_parse()
    
    if use_mmseqs:
        generate_with_mmseqs(fasta_file, dbinp, dbout, evalue, iterations, sensitivity, num_thread, pssm_dir=pssm_dir)
    else:
        generate_pssm(num_thread, fasta_dir, pssm_dir, dbinp, dbout, num, fasta_file, iterations, possum_dir)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
