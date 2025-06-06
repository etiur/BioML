from transformers import PreTrainedModel
import torch
import random
import numpy as np
from dataclasses import dataclass
from peft import replace_lora_weights_loftq, AutoPeftModel
from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from lightning.pytorch.tuner import Tuner
from pathlib import Path
import pandas as pd
from ..deep.train_config import LLMConfig


def estimate_deepmodel_size(model: PreTrainedModel, precision: torch.dtype):
    """
    Estimate the size of the model in memory.

    Parameters
    ----------
    model : PreTrainedModel
        The pre-trained model to estimate the size of.
    precision : torch.dtype
        The precision of the model's parameters. Can be torch.float16 for half precision or torch.float32 for single precision.

    Returns
    -------
    str
        The estimated size of the model in megabytes (MB), rounded to two decimal places.
    """
    num = 2 if precision==torch.float16 else 4 # float16 takes 2 bytes and float32 takes 4 bytes per parameter
    size = round(model.num_parameters() * num/1000_000, 2)
    return f"{size} MB"


def print_trainable_parameters(model: PreTrainedModel):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
def set_seed(seed: int):
    """
    Set the seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        The seed value to set.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True # use deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    
def load_adapter(peft_model: str, llm_config: dataclass = LLMConfig(),
                 adapters: dict[str, str] | None=None, use_adapter: str="initial"):
    """
    Load a PEFT model with the specified adapter.

    Parameters
    ----------
    peft_model : str
        The path to the saved peft model (the lora weights for example).
    llm_config : dataclass, optional
        The configuration for the LLM, by default LLMConfig()
    adapters : dict, optional
        A dictionary of additional adapters to load, by default None
    use_adapter : str, optional
        The name of the adapter to use, by default "initial"

    Returns
    -------
    PreTrainedModel
        The loaded PEFT model with the specified adapter.
    """
    # Load the PEFT model with the specified adapter
    device = "auto" if llm_config.device == "cuda" else llm_config.device
    # it shows something like not initialized but that is fine
    model = AutoPeftModel.from_pretrained(peft_model, adapter_name=use_adapter, 
                                          low_cpu_mem_usage=True, device_map=device,
                                          torch_dtype=llm_config.dtype)                                                                
    if adapters:
        for key, value in adapters.items():
            model.load_adapter(value, adapter_name=key)
    model.set_adapter(use_adapter)
    model.merge_adapter()
    return model

def get_mae(x, y):
    return (x - y).abs().mean()

def get_mse(x, y):
    return torch.pow(x - y, 2).mean()


def loftq_initialization(model, inputs):
    
    logits_base = model(input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"]).logits
    
    def my_callback(model, module_name):
        """Callable to replace weights with LoFTQ if the mse is lower than the current best one."""
        logits = model(**inputs).logits
        mse = get_mse(logits_base, logits)
        current_mse = float("inf")
        if mse < current_mse:
            current_mse = mse
            print(f"MSE improved for module {module_name}")
            return True
        print(f"MSE did not improve for module {module_name}")
        return False
    
    replace_lora_weights_loftq(model, callback=my_callback)


def learning_rate_finder(trainer: Trainer, light_mod: LightningModule, 
                  data_module: LightningDataModule,
                  output_dir: str | Path, 
                  random_numer: int =52324, **tuner_kwargs):
    
    seed_everything(random_numer, workers=True)
    tuner = Tuner(trainer)
    # Run learning rate finder
    lr_finder = tuner.lr_find(light_mod, data_module, max_lr=1, min_lr=1e-6, **tuner_kwargs)
    # Results can be found in
    pd.DataFrame(lr_finder.results).to_csv(output_dir/"lr_finder_results.csv")
    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.savefig(output_dir/"lr_finder_plot.png")

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    return new_lr, lr_finder