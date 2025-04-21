from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
import torch
from datasets import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from torch.utils.data import DataLoader
import argparse
from .train_config import LLMConfig
from ..utilities.deep_utils import set_seed
from ..utilities.utils import convert_to_parquet, load_config


def arg_parse():
    parser = argparse.ArgumentParser(description="Generate embeddings from the protein large language model in Huggingface")
    parser.add_argument("fasta_file", type=str, help="Path to the FASTA file")
    parser.add_argument("-m", "--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", 
                        help="Name of the language model from huggingface")
    parser.add_argument("-d", "--mode", default="append", choices=("append", "write"),
                        help="Whether to write all the embeddings at once or one batch at the time")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="The batch size")
    parser.add_argument("-p", "--save_path", type=str, default="embeddings.csv", help="The path to save the emebeddings in csv format")
    parser.add_argument("-s","--seed", type=int, default=63462634, help="Seed for reproducibility")
    parser.add_argument("-op", "--option", type=str, default="mean", help="Option to concatenate the embeddings", 
                        choices=("mean", "sum", "max", "flatten"))
    parser.add_argument("-f", "--format", type=str, default="csv", choices=("csv", "parquet"), 
                        help="Format to save the embeddings")
    parser.add_argument("-l", "--llm_config", type=str, default="",
                        help="Path to the language model configuration file (optional). json or yaml file.")
    parser.add_argument("-t", "--tokenizer_args", type=str, default="",
                        help="Path to the tokenizer configuration file (optional). json or yaml file.")
    parser.add_argument("-pt", "--pretrained_args", type=str, default="",
                        help="Path to the pretrained model configuration file (optional), used in the AutoModel.from_pretrained function. json or yaml file.")
    args = parser.parse_args()
    return [args.fasta_file, args.model_name, args.batch_size, args.save_path, args.seed, args.option,
            args.format, args.mode, args.llm_config, args.pretrained_args, args.tokenizer_args]

    
@dataclass(slots=True)
class TokenizeFasta:
    """
    Tokenize the fasta file.

    Parameters
    ----------
    config : LLMConfig
        Configuration for the language model.
    tokenizer : None
        Tokenizer for the language model.
    """
    config: LLMConfig = field(default_factory=LLMConfig)
    tokenizer: None = field(default=None, init=False)
    tokenizer_args: dict = field(default_factory=dict)
     
    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, low_cpu_mem_usage=True)
        if "esm" in self.config.model_name:
            #self.tokenizer_args["add_special_tokens"] = False
            self.tokenizer_args["padding"] = True
            self.tokenizer_args["truncation"] = True

    def chunks(self, fasta_file: str):
        """
        Split the fasta file into individual examples.

        Parameters
        ----------
        fasta_file : str
            Path to the FASTA file.

        Yields
        ------
        dict[str, str]
            A sample of the fasta sequence.
        """
        with open(fasta_file, 'r') as f:
            seqs = SeqIO.parse(f, 'fasta')
            for seq in seqs:
                yield {"id":seq.id, "seq":str(seq.seq)}

    def create_dataset(self, fasta_file: str):
        """
        Create a dataset from the fasta file.

        Parameters
        ----------
        fasta_file : str
            Path to the FASTA file.

        Returns
        -------
        Dataset
            Dataset of the fasta sequences.
        """
        return Dataset.from_generator(self.chunks, gen_kwargs={"fasta_file": fasta_file})

    def tokenize(self, fasta_file: str, add_columns: tuple=(), 
                 removes_column: str | list[str]=[]) -> Dataset:
        """
        Tokenize the batch of sequences.

        Parameters
        ----------
        batch_seq : dict[str, str]
            Batch of sequences.
        add_columns : tuple[list]
            Additional columns to add to the tokenized sequences. Tuple of list[column_name, column_values].
        removes_column : str | list[str]
            Columns to remove from the tokenized sequences.

        Returns
        -------
        Dataset
            Tokenized sequences.
        """
        dataset = self.create_dataset(fasta_file)
        cols = []
        tok = dataset.map(lambda examples: self.tokenizer(examples["seq"], return_tensors="np", **self.tokenizer_args),
                          batched=True)
        tok.set_format(type="torch", columns=["input_ids", "attention_mask"], device=self.config.device)
        for x in add_columns:
            tok = tok.add_column(x[0], x[1])
            cols.append(x[0])
        if removes_column:
            tok = tok.remove_columns(removes_column)
        return tok


@dataclass(slots=True)
class ExtractEmbeddings:
    """
    Extract embeddings from the tokenized sequences.

    Parameters
    ----------
    config : LLMConfig
        Configuration for the language model.
    model : None
        Language model to use for the embeddings.

    """
    config: LLMConfig = field(default_factory=LLMConfig)
    model: None = field(default=None, init=False)
    pretrained_args: dict = field(default_factory=dict)

    def __post_init__(self):

        device = "auto" if self.config.device == "cuda" else self.config.device
        if "esm" in self.config.model_name:
            self.pretrained_args["add_pooling_layer"] = False
        self.model = AutoModel.from_pretrained(self.config.model_name, output_hidden_states=True, device_map=device, 
                                               torch_dtype=self.config.dtype,
                                               low_cpu_mem_usage=True, offload_folder="offload", **self.pretrained_args)
        self.model.eval()
        
    @staticmethod
    def concatenate_options(embedings: torch.Tensor, option: str = "mean"):
        """
        Concatenate the per residue tokens into a per sequence token

        Parameters
        ----------
        embedings : list[torch.Tensor]
            List of embeddings from the different layers.
        option : str
            Option to concatenate the embeddings.

        Returns
        -------
        torch.Tensor
            Concatenated embeddings.
        """
        if option == "mean":
            return torch.mean(embedings, dim=0)
        if option == "sum":
            return torch.sum(embedings, dim=0)
        if option == "max":
            return torch.max(embedings, dim=0)[0]
        if option == "flatten":
            return torch.flatten(embedings)
        else:
            raise ValueError("Option not available yet. Choose between 'mean', 'sum', 'max' or 'flatten'")
    
    def extract(self, batch_seq_keys: list[str], tok: dict[str, torch.Tensor], 
                option: str = "mean"):
        """
        Extract embeddings from the tokenized sequences.

        Parameters
        ----------
        batch_seq_keys : list[str]
            Identifiers or index for the sequences within the batch.
        tok : dict[str, torch.Tensor]
            Tokenized sequences.
        option : str
            Option to concatenate the embeddings.
        
        Returns
        -------
        dict[str, np.array]
            Extracted embeddings.
        """
        results = {}
        with torch.no_grad():
            output = self.model(**tok)
        if self.config.hidden_state_to_extract == -1:
            try:
                hidden_states = output.last_hidden_state
            except AttributeError:
                hidden_states = output.hidden_states[-1]
        else:
            hidden_states = output.hidden_states[self.config.hidden_state_to_extract]
            
        mask = tok["attention_mask"].bool()
        for num, x in enumerate(hidden_states):
            masked_x = x[mask[num]]
            results[batch_seq_keys[num]] = self.concatenate_options(masked_x, 
                                                                    option).detach().cpu().numpy()
        return results

    @staticmethod
    def save(results: dict[str, np.array], path: str | Path, mode: str="a"):
        """
        Save the embeddings to a CSV file.

        Parameters
        ----------
        results : dict[str, np.array]
            Embeddings to save.
        path : str
            Path to the CSV file.
        """
        if mode == "a":
            embeddings = pd.DataFrame(results).T
        else:
            embeddings = pd.concat([pd.DataFrame(res).T for res in results])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        embeddings.to_csv(path, mode=mode, header=not Path(path).exists())
    
    
    def batch_extract_save(self, seq_keys: list[str], dataset: Dataset, batch_size: int=8, 
                           save_path: str | Path = "embeddings.csv", option: str = "mean",
                           format_: str = "csv", mode: str="append"):
        """
        Extract and save embeddings from a batch of sequences.

        Parameters
        ----------
        seq_keys : list[str]
            Keys for the sequences.
        dataset : Dataset
            Tokenized sequences saved in a Dataset object from Huggingface.
        batch_size : int, optional
            The batch size, by default 8
        save_path : str, optional
            The path to save the emebeddings in csv format, by default "embeddings.csv"
        option : str, optional
            Option to concatenate the embeddings, by default "mean"
        format_ : str, optional
            Format to save the embeddings, by default "csv" but can also be parquet
        Note
        ----
        This function saves at each batch iteration because it is thought for large files
        that doesn't fit in-memory.
        """
        save_path = Path(save_path)
        res = []
        for num, batch in enumerate(DataLoader(dataset, batch_size=batch_size)):
            batch_seq_keys = seq_keys[num*batch_size:(num+1)*batch_size]
            results = self.extract(batch_seq_keys, batch, option)
            if mode == "append":
                self.save(results, save_path, mode="a")
            else:
                res.append(results)
        if mode == "write":
            self.save(res, save_path, mode="w")

        if format_ == "parquet":
            convert_to_parquet(save_path, save_path.with_suffix(".parquet"))


def generate_embeddings(fasta_file: str, model_name: str="facebook/esm2_t6_8M_UR50D",
                        llm_args: dict = dict(), pretrained_args: dict=dict(), tokenizer_args: dict=dict(),
                        batch_size: int=8, save_path: str = "embeddings.csv", 
                        option: str = "mean", format_: str = "csv", mode: str="append"):
    """
    Generate embeddings from a FASTA file.

    Parameters
    ----------
    fasta_file : str
        The fasta file to tokenize
    model_name : str
        The protein language model to use from Huggingface, by default "facebook/esm2_t6_8M_UR50D"
    llm_args : dict, optional
        Arguments for the LLMConfig, by default {}
    pretrained_args : dict, optional
        Arguments for the pretrained model, by default {}
    tokenizer_args : dict, optional
        Arguments for the tokenizer, by default {}
    batch_size : int, optional
        The batch size, by default 8
    save_path : str, optional
        The path to save the emebeddings in csv format, by default "embeddings.csv"
    option : str, optional
        Option to concatenate the embeddings, by default "mean"
    format_ : str, optional
        Format to save the embeddings, by default "csv" but can also be parquet
    """

    config = LLMConfig(model_name, **llm_args)
    tokenizer = TokenizeFasta(config, tokenizer_args=tokenizer_args)
    embeddings = ExtractEmbeddings(config, pretrained_args=pretrained_args)
    tok = tokenizer.tokenize(fasta_file)

    # even if I have more columns in tok, it will only get the input_ids and the attention_mask
    embeddings.batch_extract_save(tok["id"], tok, batch_size, save_path, option, format_, mode=mode)


def main():
    fasta_file, model_name, batch_size, save_path, seed, option, format, mode, llm_args, pretrained_args, tokenizer_args = arg_parse()
    set_seed(seed)
    tokenizer_args = load_config(tokenizer_args, extension=tokenizer_args.split(".")[-1])
    llm_args = load_config(llm_args, extension=llm_args.split(".")[-1])
    pretrained_args = load_config(pretrained_args, extension=pretrained_args.split(".")[-1])
    
    generate_embeddings(fasta_file, model_name, llm_args=llm_args, pretrained_args=pretrained_args,
                        tokenizer_args=tokenizer_args, batch_size=batch_size, save_path=save_path, 
                        option=option, format_=format, mode=mode)


if __name__ == "__main__":
    main()