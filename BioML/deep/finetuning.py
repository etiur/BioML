from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
import torch
import numpy as np
from Bio import SeqIO
from datasets import Dataset, DatasetDict
import argparse
from typing import Iterable
from sklearn.model_selection import train_test_split
from BioML.utilities import split_methods as split
from embeddings import LLMConfig, TokenizeFasta
from dataclasses import dataclass
from peft import get_peft_model, LoraConfig
from typing import Iterable
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

    args = parser.parse_args()
    return [args.fasta_file, args.model_name, args.disable_gpu, args.batch_size, args.save_path, args.seed]




def get_split(dataset: Dataset, cluster, num_split, shuffle=True, seed=None, option="cluster", stratify=True, test_size=0.2):
    if option == "cluster":
        cluster = split.ClusterSpliter(cluster, num_split, shuffle=shuffle, random_state=seed)
        train, test = cluster.train_test_split(range(len(dataset)), index=dataset["id"])
    elif option == "random":
        stratify = dataset["labels"] if stratify else None
        train, test = train_test_split(range(len(dataset)), stratify=stratify, test_size=test_size)
    return train, test

def get_data(dataset: Dataset, train_indices: Iterable[int], test_indices: Iterable[int]):
    training, testing = dataset.select(train_indices), dataset.select(test_indices)
    return training, testing
    

def get_datasetdict(train, test, validation):
     data = DatasetDict({"train":train, "test":test, "validation": validation})
    




@dataclass(slots=True)
class PEFTSetter:
    model: PreTrainedModel
    
    def get_target_module_names_for_peft(self, filter_: str | Iterable[str] ="attention"):
        """
        Get the target module names for the LoraConfigs target module option. 
        It will look if the names in target modules matches the end of the layers names or 
        it can be an exact match

        Parameters
        ----------
        model : Hugging Face model
            The model to get the target modules from
        filter_ : str | Iterable[str], optional
            Filter the names that are returned, by default "attention"

        Returns
        -------
        list[str]
            List of the target module names
        """
        if isinstance(filter_, str):
            filter_ = [filter_] # if it is a string, convert it to a list
        module_names = []
        for name, module in self.model.named_modules():
            n = name.split(".")
            if filter_ and set(n).intersection(filter_):
                module_names.append(name)
            elif not filter_:
                module_names.append(name)
        return module_names
    
    def get_lora_model(self, rank: int, target_modules: str | list[str], 
                       lora_alpha: int | None=None):
        
        if lora_alpha is None:
            lora_alpha = rank * 2
        else:
            print("Warning lora_alpha is set to a value. For optimal performance, it is recommended to set it double the rank")
        
        # get the lora models
        peft_config = LoraConfig(inference_mode=False, r=rank, lora_alpha=lora_alpha, lora_dropout=0.1, 
                                 target_modules=target_modules)
        
        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()
        return model
