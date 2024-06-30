from dataclasses import dataclass, field
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
    # training params
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


@dataclass(slots=True)
class TrainConfig:
    num_classes: int = 2 # classification default
    _objective: str = "classification"
    classi_metrics_threshold: float = 0.5
    disable_gpu: bool =False
    #lora params
    qlora: bool = False
    lora_rank: int = 64
    use_dora: bool = True
    lora_alpha: int | None = None
    target_modules: list[str] | str = field(default_factory=lambda: ['query', 'key', 'value', 'attention.dense'])
    lora_dropout: float = 0.05
    modules_to_save: list[str] | str = field(default_factory=lambda: ["classifier.dense", "classifier.out_proj"])
    adapter_output: str = "peft_model"
    # lightning trainer params
    root_dir: str ="."
    deterministic: bool = False
    weight_decay: float = 0.01
    model_checkpoint_dir: str = "model_checkpoint"
    accumulate_grad_batches: int = 1
    debug_mode_sample: bool = False
    max_time: dict[str, int] | None | str = None
    batch_size: int = 8
    max_epochs: int = 20
    _precision: str = "32-true"
    # callback params
    patience: int = 4
    min_delta: float = 0.005
    optimize: str = "Val_MCC"
    optimize_mode: str = "max"
    # logging params
    log_save_dir: str = "LLM_run"
    mlflow_description: str = f"PEFT tune"
    mlflow_run_name: str = f"{uuid.uuid4().hex[:10]}"
    
    @property
    def objective(self):
        """
        Get the device to use for the language model.

        Returns
        -------
        str
            Device to use for the language model.
        """
        if self.num_classes == 1:
            return "regression"
        return "classification"
    
    @property
    def precision(self):
        """
        Get the device to use for the language model.

        Returns
        -------
        str
            Device to use for the language model.
        """
        if torch.cuda.is_available() and not self.disable_gpu:
            return "16-mixed"
        return self._precision
    
    @property
    def mlflow_experiment_name(self):
        """
        Get the device to use for the language model.

        Returns
        -------
        str
            Device to use for the language model.
        """
        if self.num_classes == 1:
            return "Mlflow Regression"
        return "Mlflow Classification"
    
    @property
    def csv_experiment_name(self):
        """
        Get the device to use for the language model.

        Returns
        -------
        str
            Device to use for the language model.
        """
        if self.num_classes == 1:
            return "CSV Regression"
        return "CSV Classification"
    
    @precision.setter
    def precision(self, value: str):
        """
        Set the device to use for the language model.

        Parameters
        ----------
        value : str
            Device to use for the language model.
        """
        self._precision = value


@dataclass(slots=True)
class SplitConfig:
    random_seed: int = 42
    stratify: bool = False
    splitting_strategy: str = "random"
    num_split: int = 5
    stratify: bool = True
    shuffle: bool = True
    test_size: float = 0.2
    cluster_file: str | None = None