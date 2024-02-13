from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
import torch
from datasets import Dataset


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
    
    def tokenize(self, seq: str | list[str]):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tok = self.tokenizer(seq, padding=True, truncation=True, return_tensors="pt", is_split_into_words=False)
        for key, value in tok.items():
            tok[key] = value.to(self.device)
        return tok
    
    def extract(self, tok: dict[str, torch.Tensor]):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
