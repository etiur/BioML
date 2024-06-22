from dataclasses import dataclass
import torch
import uuid

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
    # model params
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    disable_gpu: bool = False
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    num_classes: int = 2 # classification default
    # training params
    max_epochs: int = 10
    hidden_state_to_extract: int = -1
    
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
    
    @device.setter
    def device(self, value: str):
        """
        Set the device to use for the language model.

        Parameters
        ----------
        value : str
            Device to use for the language model.
        """
        self._device = value


@dataclass
class TrainConfig:
    fasta_file: str
    num_classes: int = 2 # classification default
    qlora: bool = False
    objective: str = "classification" if num_classes >= 2 else "regression"
    # lightning trainer params
    model_checkpoint_dir: str = "model_checkpoint"
    accumulate_grad_batches: int = 1
    debug_mode_sample: bool = False
    max_time: dict[str, int] | None | str = None
    batch_size: int = 8
    max_epochs: int = 10
    precision: str = "16-mixed" if torch.cuda.is_available() else "32-true"
    # callback params
    patience: int = 4
    min_delta: float = 0.005
    optimize: str = "Val_MCC" if num_classes >= 2 else "Val_R2"
    optimize_mode: str = "max"
    # mlflow params
    mlflow_experiment_name: str = "classification experiment" if num_classes >= 2 else "regression experiment"
    mlflow_save_dir: str = "LLM_run"
    mlflow_description: str = f"PEFT tune in {mlflow_experiment_name}."
    mlflow_run_name: str = f"{uuid.uuid4().hex[:10]}"
    




@dataclass
class SplitConfig:
    random_seed: int = 42
    stratify: bool = True
    splitting_strategy: str = "random"
    num_split: int = 5
    stratify: bool = True
    shuffle: bool = True
    test_size: float = 0.2
    cluster_file: str | None = None