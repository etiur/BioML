from dataclasses import dataclass
import torch


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
    dtype = torch.float32
    num_classes: int = 2 # classification default
    objective: str = "classification" if num_classes >= 2 else "regression"
    # training params
    batch_size: int = 8
    max_epochs: int = 10
    patience: int = 4
    min_delta: float = 0.005
    mlflow_experiment_name: str = "classification experiment" if num_classes >= 2 else "regression experiment"
    mlflow_save_dir: str = "LLM run"
    mlflow_description: str = (
        f"PEFT tune {model_name} in {mlflow_experiment_name}."
    )
    model_checkpoint_dir: str = "model_checkpoint"
    accumulate_grad_batches: int = 1
    debug_mode_sample: bool = False
    max_time: int | None = None
    ligthning_model_root_dir: str = "lightning_model"
    
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
