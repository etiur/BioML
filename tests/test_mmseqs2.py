from pathlib import Path
from subprocess import Popen, PIPE
from collections import defaultdict
import shlex
import time

def run_program_subprocess(commands: list[str], program_name: str | None=None, 
                           shell: bool=False):
    """
    Run in parallel the subprocesses from the command
    Parameters
    ----------
    commands: list[str]
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


class MmseqsClustering:
    @classmethod
    def create_database(cls, input_file, output_database):
        """
        _summary_

        Parameters
        ----------
        input_file : _type_
            _description_
        output_database : _type_
            _description_
        """
        input_file = Path(input_file)
        output_index = Path(output_database)
        output_index.parent.mkdir(exist_ok=True, parents=True)
        command = f"mmseqs createdb {input_file} {output_index}"
        run_program_subprocess(command, "createdb")

    @classmethod
    def index_database(cls, database):
        """
        Index the target database if it is going to be reused for search 
        frequently. This will speed up the search process because it loads in memory.

        Parameters
        ----------
        database : _type_
            _description_
        """
        database = Path(database)
        command = f"mmseqs createindex {database} tmp"
        run_program_subprocess(command, "create index")
    
    @classmethod
    def cluster(cls, database, cluster_tsv="cluster.tsv", cluster_at_sequence_identity=0.3, 
                sensitivity=6.5, **cluster_kwargs):
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
    def generate_pssm(cls, query_db, search_db, evalue=0.01, num_iterations=3, pssm_filename="result.pssm", max_seqs=600, 
                      sensitivity=6.5, **search_kwags):
        search = f"mmseqs search {query_db} {search_db} result.out tmp -e {evalue} --num-iterations {num_iterations} --max-seqs {max_seqs} -s {sensitivity} -a"
        for key, value in search_kwags.items():
            search += f" --{key} {value}"
        run_program_subprocess(search, "search")
        profile = f"mmseqs result2profile {query_db} {search_db} result.out result.profile"
        run_program_subprocess(profile, "generate_profile")
        pssm = f"mmseqs profile2pssm result.profile {pssm_filename}"
        run_program_subprocess(pssm, "convert profile to pssm")

    @classmethod
    def read_cluster_info(cls, file_path):
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
    def easy_cluster(cls, input_file, cluster_tsv, cluster_at_sequence_identity=0.3, sensitivity=6.5, **cluster_kwargs):
        query_db = Path(input_file).with_suffix("")/"querydb"
        if not query_db.exists():
            cls.create_database(input_file, query_db)
        cls.cluster(query_db, cluster_tsv, cluster_at_sequence_identity, sensitivity, **cluster_kwargs)
        return cls.read_cluster_info(cluster_tsv)

    @classmethod
    def easy_generate_pssm(cls, input_file, database_file, evalue=0.01, num_iterations=3, sensitivity=6.5,
                           pssm_filename="result.pssm", generate_searchdb=False, **search_kwags):
        
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
                          sensitivity, **search_kwags)
        return pssm_filename
    
    @classmethod
    def split_pssm(cls, pssm_filename: str | Path, output_dir: str | Path ="pssm"):
        cls.write_pssm(cls.iterate_pssm(pssm_filename), output_dir)

    @classmethod
    def iterate_pssm(cls, pssm_filename: str | Path):
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
    def write_pssm(cls, pssm_tuple, output_dir: str | Path ="pssm"):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        for key, value in pssm_tuple:
            hold = ["\n"]
            hold.extend(value)
            with open(f"{output_dir}/pssm_{key}.pssm", "w") as f:
                f.writelines(hold)


#MmseqsClustering.easy_generate_pssm("no_short.fasta", "databases/uniref50")
MmseqsClustering.split_pssm("result.pssm")
