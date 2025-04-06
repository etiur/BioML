"""
Suggest mutations using a protein large language model from Huggingface.
"""

from transformers import AutoTokenizer, EsmForMaskedLM, pipeline
import torch
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Callable
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO
import argparse
from .train_config import LLMConfig
from ..utilities.utils import convert_to_parquet, load_config

def arg_parse():
    parser = argparse.ArgumentParser(description="Generate embeddings from the protein large language model in Huggingface")
    parser.add_argument("fasta_file", type=str, help="Path to the FASTA file")
    parser.add_argument("-m", "--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", 
                        help="Name of the language model from huggingface")
    parser.add_argument("-p", "--save_path", type=str, default="suggestions.csv", 
                        help="The path to save the probabilities in csv format")
    parser.add_argument("-l", "--llm_config", type=str, default="",
                        help="Path to the language model configuration file (optional). json or yaml file.")
    parser.add_argument("-t", "--tokenizer_args", type=str, default="",
                        help="Path to the tokenizer configuration file (optional). json or yaml file.")
    parser.add_argument("-pt", "--pretrained_args", type=str, default="",
                        help="Path to the pretrained model configuration file (optional), used in the AutoModel.from_pretrained function. json or yaml file.")
    parser.add_argument("-s", "--strategy", type=str, default="masked_marginal", choices=["masked_marginal", "wild_marginal"],
                        help="The strategy to use for the probabilities. masked_marginal or wild_marginal")
    parser.add_argument("-pos", "--positions", type=int, nargs="+", default=(),
                        help="The positions to get the probabilities for. If not provided, all positions will be used.")
    args = parser.parse_args()
    return [args.fasta_file, args.model_name, args.save_path,
            args.llm_config, args.pretrained_args, args.tokenizer_args, args.strategy, args.positions]


def masked_marginal(positions: Sequence[int], input_ids: torch.Tensor, tokenizer: AutoTokenizer, 
                    model: AutoModel) -> dict[int, torch.Tensor]:
    """
    Get the probabilities using the masked marginal
    where the input sequence is masked at the positions of interest and then the probabilities are calculated.
    
    Parameters
    ----------
    positions : Sequence[int]
        The positions to get the probabilities for.
    input_ids : torch.Tensor    
        The input ids of the sequence.
    tokenizer : AutoTokenizer
        The tokenizer to use for the sequence.
    model : AutoModel
        The model to use for the sequence.
    Returns
    -------
    dict[int, torch.Tensor]
        A dictionary with the positions as keys and the probabilities as values.
    """
    all_prob = {}
    for x in positions:
        masked_input_ids = input_ids.clone()
        # The plus + makes sure we are assigning the correct positions to the correct index (since CLS is at the begining of teh position)
        masked_input_ids[0, x+1] = tokenizer.mask_token_id
        with torch.no_grad():
            output = model(masked_input_ids).logits # to remove the probabilities of the tokens [CLS] and [SEP]
        probabilities = torch.nn.functional.softmax(output[0, x+1], dim=0)
        all_prob[x] = torch.log(probabilities)
    return all_prob

def wild_marginal(positions: Sequence[int], input_ids: torch.Tensor, tokenizer: AutoTokenizer, 
                  model: AutoModel) -> dict[int, torch.Tensor]:
    """
    Get the probabilities using the wild type marginal.
    where there is no masking of the input sequence.
    Parameters
    ----------
    positions : Sequence[int]
        The positions to get the probabilities for.
    input_ids : torch.Tensor    
        The input ids of the sequence.
    tokenizer : AutoTokenizer
        The tokenizer to use for the sequence.
    model : AutoModel
        The model to use for the sequence.
    Returns
    -------
    dict[int, torch.Tensor]
        A dictionary with the positions as keys and the probabilities as values.
    """
    all_prob = {}
    with torch.no_grad():
	    output = model(input_ids).logits
     for x in positions:
        # softmaxing the probabilities of the correct positions -> so it is shape 33 the probabilities
        probabilities = torch.nn.functional.softmax(output[0, x+1], dim=0)
        all_prob[x] = torch.log(probabilities)
    return all_prob

@dataclass
class SuggestMutations:
    """
    Generateper residue probabilities from the tokenized sequences.

    Parameters
    ----------
    config : LLMConfig
        Configuration for the language model.
    model : None
        Language model to use for the embeddings.
    pretrained_args : dict
        Arguments for the pretrained model.
    tokenizer_args : dict
        Arguments for the tokenizer.
    """
    config: LLMConfig = field(default_factory=LLMConfig)
    pretrained_args: dict = field(default_factory=dict)
    tokenizer_args: dict = field(default_factory=dict)
    model: None = field(default=None, init=False)
    tokenizer: None = field(default=None, init=False)
    
    def __post_init__(self):

        device = "auto" if self.config.device == "cuda" else self.config.device
        if "esm" in self.config.model_name:
            self.pretrained_args["add_pooling_layer"] = False
        self.model = AutoModel.from_pretrained(self.config.model_name, output_hidden_states=True, device_map=device, 
                                                torch_dtype=self.config.dtype,
                                                low_cpu_mem_usage=True, offload_folder="offload", **self.pretrained_args)
        
        if "esm" in self.config.model_name:
            self.tokenizer_args["padding"] = True
            self.tokenizer_args["truncation"] = True
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, low_cpu_mem_usage=True, **self.tokenizer_args)
            
    def get_probabilities(self, protein_sequence: str, positions: Sequence[int]=(), 
                         strategy: Callable=masked_marginal):
        """
        Get the probabilities of the amino acids in the protein sequence.
        
        """
        # Encode the protein sequence
        input_ids = self.tokenizer.encode(protein_sequence, return_tensors="pt") # it will add a cls and eos tokens, so the lenght is less 
        # sequence_length = input_ids.shape[1] - 2 
        # List of amino acids
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        if not isinstance(amino_acids, str):
            raise TypeError("amino_acids should be a string")
        aa_ids = {aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids}
        if not aa_ids:
            raise ValueError("Could not convert tokens to ids")
        prob_mt = {}
        # Get the probabilities	
        all_prob = strategy(positions, input_ids, self.tokenizer, self.model)
        if not all_prob:
            raise ValueError("Could not get the probabilities")
        if not positions:
            positions = range(input_ids.shape[1]-2) # -2 to remove the CLS and EOS tokens
            # This will get he probabilities of all the positions in the sequence
            
        for pos in positions:
            wt_residue_id = input_ids[0, pos+1].item()
            wt_token = self.tokenizer.convert_ids_to_tokens(wt_residue_id)
            # Get the probability of the wild type residue
            prob_wt = all_prob[pos][wt_residue_id].item()
            # Get the probability of the mutant residue relative to the wild type residue
            prob_mt[f"{wt_token}{pos}"] = {f"{key}": all_prob[pos][value].item() - prob_wt for key, value in aa_ids.items()}

        
        suggestions = pd.DataFrame(prob_mt).T
        return suggestions
    
    def read_fasta(self, fasta_file: str | Path) -> dict[str, str]:
        """
        Read a FASTA file and yield the sequences.
        
        Parameters
        ----------
        fasta_file : str | Path
            The path to the FASTA file.
        
        Returns
        -------
        dict[str, str]
            A dictionary with the sequence id and the sequence.
        """
        with open(fasta_file, 'r') as f:
            seqs = list(SeqIO.parse(f, 'fasta'))

        return {str(seq.id): str(seq.seq) for seq in seqs} # return a dictionary with the id and the sequence


    def get_probabilities_from_fasta(self, fasta_file: str | Path, positions: Sequence[int]=(), 
                                     strategy: Callable=masked_marginal, save_path: str | Path="suggestions.csv"):
        """
        Get the probabilities of the amino acids in the protein sequence.
        """ 
        # Read the FASTA file and get the sequences
        sequences = self.read_fasta(fasta_file)
        # Get the probabilities for each sequence
        all_prob = {}
        for seq_id, protein_sequence in sequences.items():
            all_prob[seq_id] = self.get_probabilities(protein_sequence, positions, strategy)
        # Save the probabilities to a file
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_prob).to_csv(save_path)   
        return pd.concat(all_prob)


def main():
    fasta_file, model_name, save_path, llm_config, pretrained_args, tokenizer_args, strategy, positions = arg_parse()
    # Load the configuration file if provided
    tokenizer_args = load_config(tokenizer_args, extension=tokenizer_args.split(".")[-1])
    llm_args = load_config(llm_args, extension=llm_args.split(".")[-1])
    pretrained_args = load_config(pretrained_args, extension=pretrained_args.split(".")[-1])
    # Create the configuration object
    config = LLMConfig(model_name=model_name, **llm_args)
    # Create the SuggestMutations object
    stra = {"masked_marginal": masked_marginal, "wild_marginal": wild_marginal}
    suggest_mutations = SuggestMutations(config, pretrained_args=pretrained_args, tokenizer_args=tokenizer_args)
    suggest_mutations.get_probabilities_from_fasta(fasta_file, save_path=save_path, 
                                                   strategy=stra[strategy], positions=positions)
    print(f"Probabilities saved to {save_path}")