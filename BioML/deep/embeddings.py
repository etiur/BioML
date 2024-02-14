from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
import torch
from datasets import Dataset
import random
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import islice


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True # use deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(slots=True)
class ExtractEmbeddings:
    disbale_gpu: bool = False
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer: None = field(default=None, init=False)
    model: None = field(default=None, init=False)

    def __post_init__(self):
        if self.disbale_gpu:
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)

    def chunks(self, seq: dict[str,str], size: int=10):
        """
        Split the dictionary into chunks of the given size.

        Parameters
        ----------
        seq : dict[str, str]
            Dictionary of sequences.
        size : int, optional
            Size of the chunks, by default 10.

        Yields
        ------
        dict[str, str]
            Chunk of the sequence.
        """
        it = iter(seq.items())
        for _ in range(0, len(seq), size):
            yield dict(islice(it, size))

    def tokenize(self, batch_seq: dict[str,str]):
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
        tok = self.tokenizer(batch_seq.values(), padding=True, truncation=True, return_tensors="pt", is_split_into_words=False)
        for key, value in tok.items():
            tok[key] = value.to(self.device)
        return tok
    
    def extract(self, batch_seq_keys: str, tok: dict[str, torch.Tensor]):
        """
        Extract embeddings from the tokenized sequences.

        Parameters
        ----------
        batch_seq_keys : str
            Keys of the batch of sequences.
        tok : dict[str, torch.Tensor]
            Tokenized sequences.

        Returns
        -------
        dict[str, np.array]
            Extracted embeddings.
        """
        results = {}
        output = self.model(**tok)
        mask = tok["attention_mask"].bool()
        for num, x in enumerate(output.hidden_states[-1]):
            masked_x = x[mask[num]]
            results[batch_seq_keys[num]] = masked_x.mean(dim=0).detach().cpu().numpy()
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

