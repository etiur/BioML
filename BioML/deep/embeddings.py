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
from ..utilities.utils import set_seed


def arg_parse():
    parser = argparse.ArgumentParser(description="Generate embeddings from the protein large language model in Huggingface")
    parser.add_argument("fasta_file", type=str, help="Path to the FASTA file")
    parser.add_argument("-m", "--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", 
                        help="Name of the language model from huggingface")
    parser.add_argument("-d", "--disable_gpu", action="store_true", help="Whether to disable the GPU")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="The batch size")
    parser.add_argument("-p", "--save_path", type=str, default="embeddings.csv", help="The path to save the emebeddings in csv format")
    parser.add_argument("-s","--seed", type=int, default=12891245318, help="Seed for reproducibility")
    parser.add_argument("-op", "--option", type=str, default="mean", help="Option to concatenate the embeddings")
    parser.add_argument("-f", "--format", type=str, default="csv", choices=("csv", "parquet"), 
                        help="Format to save the embeddings")

    args = parser.parse_args()
    return [args.fasta_file, args.model_name, args.disable_gpu, args.batch_size, args.save_path, args.seed, args.option,
            args.format]



@dataclass(slots=True)
class LLMConfig:
    """
    Configuration for the language model.

    Parameters
    ----------  
    model_name : str
        Name of the language model.
    _device : str
        Device to use for the language model.
    disbale_gpu : bool
        Whether to disable the GPU.
    """
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    disable_gpu: bool = False
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self):
        """
        Get the device to use for the language model.

        Returns
        -------
        str
            Device to use for the language model.
        """
        if self.disable_gpu:
            return "cpu"
        return self._device
    

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
    config: LLMConfig
    tokenizer: None = field(default=None, init=False)
     
    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

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

    def tokenize(self, fasta_file: str):
        """
        Tokenize the batch of sequences.

        Parameters
        ----------
        batch_seq : dict[str, str]
            Batch of sequences.

        Returns
        -------
        dict[str, torch.Tensor]
            Tokenized sequences.
        """
        dataset = self.create_dataset(fasta_file)
        tok = dataset.map(lambda examples: self.tokenizer(examples["seq"], return_tensors="np", 
                          padding=True, truncation=True), batched=True)
        tok.set_format(type="torch", columns=["input_ids", "attention_mask"], device=self.config.device)
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
    config: LLMConfig
    model: None = field(default=None, init=False)

    def __post_init__(self):

        self.model = AutoModel.from_pretrained(self.config.model_name, add_pooling_layer=False, output_hidden_states=True)
        self.model.to(self.config.device)

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
        else:
            raise ValueError("Option not available yet. Choose between 'mean', 'sum' or max")
    
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
        mask = tok["attention_mask"].bool()
        for num, x in enumerate(output.last_hidden_state):
            masked_x = x[mask[num]]
            results[batch_seq_keys[num]] = self.concatenate_options(masked_x, 
                                                                    option).detach().cpu().numpy()
        return results

    @staticmethod
    def save(results: dict[str, np.array], path: str | Path):
        """
        Save the embeddings to a CSV file.

        Parameters
        ----------
        results : dict[str, np.array]
            Embeddings to save.
        path : str
            Path to the CSV file.
        """
        embeddings = pd.DataFrame(results).T 
        embeddings.to_csv(path, mode='a', header=not Path(path).exists())
    
    @staticmethod
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
    
    def batch_extract_save(self, seq_keys: list[str], dataset: Dataset, batch_size: int=8, 
                           save_path: str | Path = "embeddings.csv", option: str = "mean",
                           format_: str = "csv"):
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
        for num, batch in enumerate(DataLoader(dataset, batch_size=batch_size)):
            batch_seq_keys = seq_keys[num*batch_size:(num+1)*batch_size]
            results = self.extract(batch_seq_keys, batch, option)
            self.save(results, save_path)
        if format_ == "parquet":
            self.convert_to_parquet(save_path, save_path.with_suffix(".parquet"))


def generate_embeddings(model_name: str, fasta_file: str, disable_gpu: bool=False, 
                        batch_size: int=8, save_path: str = "embeddings.csv", 
                        option: str = "mean", format_: str = "csv"):
    """
    Generate embeddings from a FASTA file.

    Parameters
    ----------
    model_name : str
        The protein language model to use from Huggingface
    fasta_file : str
        The fasta file to tokenize
    disable_gpu : bool, optional
        Whether to disable the GPU, by default False
    batch_size : int, optional
        The batch size, by default 8
    save_path : str, optional
        The path to save the emebeddings in csv format, by default "embeddings.csv"
    option : str, optional
        Option to concatenate the embeddings, by default "mean"
    format_ : str, optional
        Format to save the embeddings, by default "csv" but can also be parquet
    """

    config = LLMConfig(model_name, disable_gpu=disable_gpu)
    tokenizer = TokenizeFasta(config)
    embeddings = ExtractEmbeddings(config)
    tok = tokenizer.tokenize(fasta_file)

    # even if I have more columns in tok, it will only get the input_ids and the attention_mask
    embeddings.batch_extract_save(tok["id"], tok, batch_size, save_path, option, format_)


def main():
    fasta_file, model_name, disable_gpu, batch_size, save_path, seed, option, format = arg_parse()
    set_seed(seed)
    generate_embeddings(model_name, fasta_file, disable_gpu, batch_size, save_path, option, format)


if __name__ == "__main__":
    main()