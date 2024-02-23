from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
import torch
from datasets import Dataset
import random
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True # use deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)


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
        dataset = Dataset.from_generator(self.chunks, gen_kwargs={"fasta_file": fasta_file})
        tok = dataset.map(lambda examples: self.tokenizer(examples["seq"], return_tensors="np", 
                          padding=True, truncation=True), batched=True)
        tok.set_format(type="torch", columns=["input_ids", "attention_mask"], device=self.config.device)
        return tok


@dataclass(slots=True)
class ExtractEmbeddings:
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
        elif option == "sum":
            return torch.sum(embedings, dim=0)
        elif option == "max":
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

    def save(self, results: dict[str, np.array], path: str):
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
    
    def batch_extract_save(self, seq_keys: list[str], dataset: Dataset, batch_size: int=8, 
                           save_path: str = "embeddings.csv", option: str = "mean"):
        """
        Extract and save embeddings from a batch of sequences.

        Parameters
        ----------
        seq_keys : list[str]
            Keys for the sequences.
        dataset : Dataset
            Tokenized sequences saved in a Dataset objectf from Huggingface.
        batch_size : int, optional
            The batch size, by default 8
        save_path : str, optional
            The path to save the emebeddings in csv format, by default "embeddings.csv"
        option : str, optional
            Option to concatenate the embeddings, by default "mean"
        
        Note
        ----
        This function saves at each batch iteration because it is thought for large files
        that doesn't fit in-memory.
        """
        for num, batch in enumerate(DataLoader(dataset, batch_size=batch_size)):
            batch_seq_keys = seq_keys[num*batch_size:(num+1)*batch_size]
            results = self.extract(batch_seq_keys, batch, option)
            self.save(results, save_path)


def generate_embeddings(model_name: str, fasta_file: str, disable_gpu: bool=False, 
                        batch_size: int=8, save_path: str = "embeddings.csv"):
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
    """

    config = LLMConfig(model_name, disable_gpu=disable_gpu)
    tokenizer = TokenizeFasta(config)
    embeddings = ExtractEmbeddings(config)
    tok = tokenizer.tokenize(fasta_file)

    # even if I have more columns in tok, it will only get the input_ids and the attention_mask
    embeddings.batch_extract_save(tok["id"], tok, batch_size, save_path)
    