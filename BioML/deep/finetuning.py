from transformers import AutoModelForSequenceClassification, PreTrainedModel, BitsAndBytesConfig
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import argparse
from typing import Iterable, Any
from sklearn.model_selection import train_test_split
from BioML.utilities import split_methods as split
from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
import bitsandbytes as bnb
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from dataclasses import dataclass, asdict, field
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from typing import Iterable
from torchmetrics.classification import (
    Accuracy, F1Score, Precision, Recall, AUROC, AveragePrecision, CohenKappa, MatthewsCorrCoef) 
from torchmetrics.regression import (
    MeanAbsoluteError, MeanSquaredError,  PearsonCorrCoef, KendallRankCorrCoef, R2Score,
    MeanAbsolutePercentageError, MeanSquaredLogError, SpearmanCorrCoef)
import numpy as np
from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, CSVLogger
import mlflow
from .embeddings import TokenizeFasta
from .train_config import LLMConfig, SplitConfig, TrainConfig
from ..utilities.utils import load_config


def parse_args():
    """
    Parse command line arguments for the training loop.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a model using a fasta file and labels.")
    
    parser.add_argument("-i", "--fasta_file", type=Path, required=True,
                        help="The path to the fasta file.")
    parser.add_argument("--label", type=str, required=True,
                        help="The path to the file containing labels that can be read by numpy load")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="The learning rate. Default is 1e-3.")
    parser.add_argument("-tr", "--train_config", type=str, default="",
                        help="Path to the training configuration file (optional). json or yaml file.")
    parser.add_argument("-lc", "--llm_config", type=str, default="",
                        help="Path to the language model configuration file (optional). json or yaml file.")
    parser.add_argument("-sp", "--split_config", type=str, default="",
                        help="Path to the splitting configuration file (optional). json or yaml file.")
    parser.add_argument("-tc", "--tokenizer_args", type=str, default="{}",
                        help="JSON string of arguments to pass to the tokenizer (optional). json or yaml file.")
    parser.add_argument("-lt", "--lightning_trainer_args", type=str, default="{}",
                        help="JSON string of arguments to pass to the lightning trainer (optional). json or yaml file. ")
    parser.add_argument("-u","--use_best_model", action="store_true",
                        help="Whether to use the best model saved with checkpoint. Default is True. json or yaml file.")
    parser.add_argument("-l", "--lora_init", type=str, default="True", 
                        help="The initialization for the Lora model. Default is True.")
    return parser.parse_args()


def classification_metrics(split: str, num_classes: int=2, threshold: float=0.5) -> dict:
    """
    Lightining metrics for classification tasks

    Parameters
    ----------
    split : str
        The split to get the metrics for (Train, Val or Test)
    num_classes : int, optional
        The number of classes, by default 2
    threshold : float, optional
        Threshold to be considered positive class, by default 0.5

    Returns
    -------
    dict
        Dictionary of the metrics
    """
    task = "multiclass"
    metrics = {f"{split}_Acc": Accuracy(num_classes=num_classes, task=task, 
                                         threshold=threshold),
                f"{split}_F1":F1Score(task=task, num_classes=num_classes, 
                                      threshold=threshold),
                f"{split}_Precision": Precision(task=task, num_classes=num_classes, 
                                                threshold=threshold),
                f"{split}_Recall": Recall(task=task, num_classes=num_classes, 
                                          threshold=threshold),
                f"{split}_MCC": MatthewsCorrCoef(num_classes=num_classes, 
                                                 threshold=threshold, task=task),
                #f"{split}_Confusion_Matrix": confusion_matrix(num_classes=num_classes, normalize="true", 
                #                                              task=task, threshold=threshold),
                f"{split}_AUROC": AUROC(num_classes=num_classes, task=task),
                f"{split}_Average_Precision": AveragePrecision(num_classes=num_classes, 
                                                               task=task),
                f"{split}_Cohen_Kappa": CohenKappa(num_classes=num_classes, task=task, 
                                                   threshold=threshold)}
    return metrics


def regression_metrics(split: str) -> dict:
    """
    Lightining metrics for regression tasks

    Parameters
    ----------
    split : str
        The split to get the metrics for (Train, Val or Test)

    Returns
    -------
    dict
        Dictionary of the metrics
    """
    
    metrics = {f"{split}_MAE": MeanAbsoluteError(),
                f"{split}_MSE": MeanSquaredError(squared=False),
                f"{split}_RMSE": MeanSquaredError(squared=True),
                f"{split}_R2": R2Score(),
                f"{split}_Pearson": PearsonCorrCoef(),
                f"{split}_Kendall": KendallRankCorrCoef(),
                f"{split}_MAPE": MeanAbsolutePercentageError(),
                f"{split}_MSLE": MeanSquaredLogError(),
                f"{split}_Spearman": SpearmanCorrCoef()}
    return metrics


@dataclass(slots=True)
class PreparePEFT:
    """
    Prepare the model for training with PEFT
    """
    train_config: dataclass = field(default_factory=TrainConfig)
    llm_config: dataclass = field(default_factory=LLMConfig)
    lora_init: str | bool = True
    
    @staticmethod    
    def get_target_module_names_for_peft(model: PreTrainedModel, filter_: str | Iterable[str] ="attention"):
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
        for name, module in model.named_modules():
            n = name.split(".")
            if filter_ and set(n).intersection(filter_):
                module_names.append(name)
            elif not filter_:
                module_names.append(name)
        return module_names
    
    @staticmethod
    def get_lora_config(rank: int, target_modules: str | list[str], lora_alpha: int | None=None, 
                        lora_dropout: float=0.05, use_dora: bool=True, lora_init: str | bool=True,
                        modules_to_save: str | list[str] = ["classifier.dense", "classifier.out_proj"]):
        """
        Get the LoraConfig for the model

        Parameters
        ----------
        rank : int
            The rank of the Lora model. The higher the more costly and could lead to overfitting, but better performance
        target_modules : str | list[str]
            The target modules to apply the Lora model to
        lora_alpha : int | None, optional
            The alpha value for the Lora model, by default None
        lora_dropout : float, optional
            The dropout value for the Lora model, by default 0.05
        use_dora : bool, optional
            Whether to use Dora, by default True (it improves on Lora)
        lora_init : str | bool, optional
            The initialization for the Lora model, by default True (pissa, loftq, random, guassian)
        modules_to_save : str | list[str], optional
            The modules to save, by default ["classifier.dense", "classifier.out_proj"], the name is the same for regression
            For Hugging Face at least

        Returns
        -------
        LoraConfig
            The LoraConfig for the model
        """
        if lora_alpha is None:
            lora_alpha = rank * 2
        else:
            print("Warning lora_alpha is set to a value. For optimal performance, it is recommended to set it double the rank")
        
        # get the lora models
        peft_config = LoraConfig(init_lora_weights=lora_init, inference_mode=False, r=rank, 
                                 lora_alpha=lora_alpha, lora_dropout=lora_dropout, 
                                 target_modules=target_modules, 
                                 use_dora=use_dora, 
                                 modules_to_save=modules_to_save)
        
        return peft_config

    def setup_model(self) -> PreTrainedModel:
        """
        Setup the model for training, you can override this method to use your own huggingface model

        Returns
        -------
        PreTrainedModel
        """
        if self.train_config.qlora:
            bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            bnb_config = None
        
        device = "cuda" if self.llm_config.device == "cuda" or self.train_config.qlora else self.llm_config.device
        model = AutoModelForSequenceClassification.from_pretrained(self.llm_config.model_name, 
                                                                num_labels=self.train_config.num_classes, 
                                                                low_cpu_mem_usage=True,
                                                                torch_dtype= self.llm_config.dtype, 
                                                                quantization_config=bnb_config,
                                                                device_map=device)
        if self.train_config.qlora:
            model = prepare_model_for_kbit_training(model)
        return model
    
    def prepare_model(self) -> PeftModel:
        """
        Prepare the model for training with PEFT

        Returns
        -------
        PeftModel
        """
        model = self.setup_model()
        peft_config = self.get_lora_config(rank=self.train_config.lora_rank, 
                                           target_modules=self.train_config.target_modules, 
                                           lora_alpha=self.train_config.lora_alpha, 
                                           lora_dropout=self.train_config.lora_dropout,
                                           use_dora=self.train_config.use_dora,
                                           lora_init=self.lora_init,
                                           modules_to_save=self.train_config.modules_to_save)
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

@dataclass
class PrepareSplit:
    """
    Prepare the data for training using custtom splitters and Datasets
    """
    cluster_file: str | None = None
    shuffle: bool = True
    random_seed: int = 21321412
    splitting_strategy: str = "random"
    num_split: int = 5
    stratify: bool | None = None
    test_size: float = 0.2
    
    def get_split_indices(self, dataset: Dataset) -> tuple:
        """
        Get the split indices for the dataset

        Parameters
        ----------
        dataset : Dataset
            The dataset to split

        Returns
        -------
        tuple
            The train and test indices
        """
        if self.splitting_strategy == "cluster":
            cluster = split.ClusterSpliter(self.cluster_file, self.num_split, 
                                           shuffle=self.shuffle, random_state=self.random_seed)
            train, test = cluster.train_test_split(range(len(dataset)), groups=dataset["id"])
        elif self.splitting_strategy == "random":
            stratify = dataset["labels"].cpu() if self.stratify else None
            train, test = train_test_split(range(len(dataset)), stratify=stratify, 
                                           test_size=self.test_size)
        return train, test

    def get_data(self, dataset: Dataset) -> tuple:
        """
        Get the data for training using the indices

        Parameters
        ----------
        dataset : Dataset
            The dataset to split

        Returns
        -------
        tuple
            The train, validation and test datasets
        """
        train_indices, test_indices = self.get_split_indices(dataset)
        train_, test = dataset.select(train_indices), dataset.select(test_indices)
        train_indices, validation_indices = self.get_split_indices(train_)
        train, validation = train_.select(train_indices), train_.select(validation_indices)
        return train, validation, test


class DataModule(LightningDataModule):
    def __init__(self, splitter: PrepareSplit, fasta_file: str | Path, label: np.array, 
                 config: LLMConfig = LLMConfig(), batch_size: int=1, tokenizer_args: dict=dict()) -> None:
        """
        Prepare the data for training

        Parameters
        ----------
        splitter : PrepareSplit
            The splitter to use for the data
        fasta_file : str | Path
            The path to the fasta file
        label : np.array
            The labels for the fasta file
        config : LLMConfig, optional
            The config for the language model, by default LLMConfig()
        batch_size : int, optional
            The batch size, by default 1
        tokenizer_args : dict, optional
            The arguments to pass to the tokenizer, by default dict()
        """
        super().__init__()
        self.splitter = splitter
        self.fasta_file = fasta_file
        self.batch_size = batch_size
        self.llm_config = config
        self.tokenizer_args = tokenizer_args 
        self.label = [("labels", label)]
    
    def prepare_data(self) -> None:
        """Tokenize the fasta file and store the dataset in the class instance."""
        tokenizer = TokenizeFasta(self.llm_config, tokenizer_args=self.tokenizer_args)
        self.dataset = tokenizer.tokenize(self.fasta_file, add_columns=self.label)
        
    def setup(self, stage: str):
        """Split the data into train, validation and test sets."""
        train, validation, test = self.splitter.get_data(self.dataset)
        self.train = train
        self.validation = validation
        self.test = test
        
    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(self.validation, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(self.test, batch_size=self.batch_size)


class TransformerModule(LightningModule):
    def __init__(
        self,
        model: PeftModel,
        train_config: dataclass,
        lr: float
    ):
        """
        Initialize the LightningModule with the model, training configuration and learning rate.

        Parameters
        ----------
        model : PeftModel
            The model to train
        train_config : dataclass
            The training configuration
        lr : float
            The learning rate
        """
        super().__init__()

        self.model = model
        self.train_config = train_config
        self.metrics = {"classification": classification_metrics, 
                        "regression": regression_metrics}[self.train_config.objective]
        if self.train_config.objective == "classification":
            self.train_metrics = self.metrics(split="Train", num_classes=self.train_config.num_classes, 
                                   threshold=self.train_config.classi_metrics_threshold)
            self.val_metrics = self.metrics(split="Val", num_classes=self.train_config.num_classes, 
                                   threshold=self.train_config.classi_metrics_threshold)
            self.test_metrics = self.metrics(split="Test", num_classes=self.train_config.num_classes,
                                      threshold=self.train_config.classi_metrics_threshold)
        elif self.train_config.objective == "regression":
            self.train_metrics = self.metrics(split="Train")
            self.val_metrics = self.metrics(split="Val")
            self.test_metrics = self.metrics(split="Test")
        
        self.lr = lr
        self.save_hyperparameters(ignore=["model", "metrics", "train_metrics", "val_metrics", "test_metrics"])
        
    def forward(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        label: list[int] | None = None):
        """
        Calculate the logits and the loss by passing inputs to the model and comparing against ground truth labels. 
        Here, all of the arguments of self.model comes from the
        SequenceClassification head from HuggingFace.

        Parameters
        ----------
        input_ids : list[int]
            The input ids
        attention_mask : list[int]
            The attention mask
        label : list[int] | None, optional
            The labels, by default None

        Returns
        -------
        dict
            The logits and the loss
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label,
        )

    def _compute_predictions(self, batch) -> tuple:
        """Helper method hosting the evaluation logic common to the validation and test steps."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            label=batch["labels"])
        # For predicting probabilities, do softmax along last dimension (by row).
        if self.train_config.objective == "classification":
            #pred = torch.argmax(torch.softmax(outputs["logits"], dim=-1), dim=1)
            pred = torch.softmax(outputs["logits"], dim=-1)
        elif self.train_config.objective == "regression":
            pred = outputs["logits"].flatten()

        return outputs, pred

    def training_step(self, batch, batch_idx):
        """Perform a training step on the batch."""
        outputs, preds = self._compute_predictions(batch)
        for metric in self.train_metrics.values():
            metric.to(preds.device)
            metric.update(preds, batch["labels"].to(preds.device))
        self.log("loss", outputs["loss"], on_epoch=True, on_step=False)

        return outputs["loss"] # the automodel has its own loss function depending on the problem
    
    def on_training_epoch_end(self):
        """Log the accumulated training metrics at the end of an epoch."""
        # Log the accumulated training metrics
        for name, metric in self.train_metrics.items():
            self.log(f'{name}', metric.compute())
            metric.reset()
            
    def validation_step(self, batch, batch_idx) -> dict[str, Any]:
        """Perform a validation step on the batch."""
        outputs, preds = self._compute_predictions(batch)
        for metric in self.val_metrics.values():
            metric.to(preds.device)
            metric.update(preds, batch["labels"].to(preds.device))
        self.log("loss", outputs["loss"], on_epoch=True, on_step=False)
        return outputs["loss"]
    
    def on_validation_epoch_end(self):
        """Log the accumulated validation metrics at the end of an epoch."""
        # Log the accumulated validation metrics
        for name, metric in self.val_metrics.items():
            self.log(f'{name}', metric.compute())
            metric.reset()

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        """Perform a test step on the batch."""
        outputs, preds = self._compute_predictions(batch)
        for metric in self.test_metrics.values():
            metric.to(preds.device)
            metric.update(preds, batch["labels"].to(preds.device))
        self.log("loss", outputs["loss"], on_epoch=True, on_step=False)
        
        return outputs["loss"]
    
    def on_test_epoch_end(self):
        """ Log the accumulated test metrics at the end of an epoch."""
        # Log the accumulated test metrics
        for name, metric in self.test_metrics.items():
            self.log(f'{name}', metric.compute())
            metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Predict the output of the model. Use as below (when you have the data_loader object):
        If not you can just use model(**batch) to get the predictions (see the forward method).
        Example:
        --------
        >>> data_loader = DataLoader(...)
        >>> model = MyModel()
        >>> trainer = Trainer()
        >>> predictions = trainer.predict(model, data_loader)

        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        if self.train_config.objective == "classification":
            pred = torch.softmax(outputs["logits"], dim=-1)
        elif self.train_config.objective == "regression":
            pred = outputs["logits"]

        return pred

    def configure_optimizers(self) -> Optimizer:
        """Configure the optimizer and the learning rate scheduler."""
        if not self.train_config.qlora:
            optim = AdamW(params=self.parameters(), lr=self.lr, weight_decay=self.train_config.weight_decay)
        else:
            optim = bnb.optim.PagedAdamW(self.parameters(), lr=self.lr, weight_decay=self.train_config.weight_decay)
        scheduler = OneCycleLR(optim, max_lr=self.lr, total_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": optim, "lr_scheduler": scheduler}


def training_loop(fasta_file: str | Path, label: np.array, lr: float=1e-3,
                  train_config: dataclass = TrainConfig(), 
                  llm_config: dataclass = LLMConfig(), split_config: dataclass=SplitConfig(), 
                  tokenizer_args: dict=dict(), 
                  lightning_trainer_args: dict = dict(),
                  use_best_model: bool = True, lora_init: bool | str = True) -> tuple[TransformerModule, DataModule, str]:
    """
    Train the model using the fasta file and the labels checkpoint the model with highest score; log that model to MLflow and
    return it

    Parameters
    ----------
    fasta_file : str | Path
        The path to the fasta file
    label : np.array
        The labels for the fasta file
    lr : float, optional
        The learning rate, by default 1e-3
    train_config : dataclass, optional
        The training configuration, by default TrainConfig()
    llm_config : dataclass, optional
        The language model configuration, by default LLMConfig()
    split_config : dataclass, optional
        The splitting configuration, by default SplitConfig()
    tokenizer_args : dict, optional
        The arguments to pass to the tokenizer, by default dict()
    lightning_trainer_args : dict, optional
        The arguments to pass to the lightning trainer, by default dict()
    use_best_model : bool, optional
        Whether to use the best model saved with checkpoint, by default True
    lora_init : bool | str, optional
        The initialization for the Lora model, by default True
        
    Returns
    -------
    tuple[TransformerModule, DataModule, str]
        The trained model, the data module and the path to the best model
    """
    seed_everything(split_config.random_seed, workers=True)
    splitter = PrepareSplit(split_config.cluster_file, split_config.shuffle, split_config.random_seed, 
                            split_config.splitting_strategy, 
                            split_config.num_split, split_config.stratify, split_config.test_size)
    
    data_module = DataModule(splitter, fasta_file, label, llm_config, train_config.batch_size, tokenizer_args)

    peft = PreparePEFT(train_config, llm_config, lora_init=lora_init)
    model = peft.prepare_model()
    light_mod = TransformerModule(model, train_config, lr=lr)

    # Set Up MLflow context manager.
    mlflow.set_experiment(train_config.mlflow_experiment_name)
    with mlflow.start_run(run_name=train_config.mlflow_run_name, description=train_config.mlflow_description) as run:
        # TODO: MLflow metrics should show epochs rather than steps on the x-axis
        
        mlf_logger = MLFlowLogger(experiment_name=mlflow.get_experiment(run.info.experiment_id).name, 
                                  tracking_uri=mlflow.get_tracking_uri(), log_model=True, 
                                  save_dir=Path(train_config.root_dir) /train_config.log_save_dir)
        csv_logger = CSVLogger(save_dir=Path(train_config.root_dir) / train_config.log_save_dir, 
                               name=train_config.csv_experiment_name)
        mlf_logger._run_id = run.info.run_id
        mlflow.log_params({k: v for k, v in asdict(train_config).items() if not k.startswith("mlflow_")})
        mlflow.log_params({k: v for k, v in asdict(split_config).items() if not k.startswith("mlflow_")})
        mlflow.log_params({k: v for k, v in asdict(llm_config).items() if not k not in asdict(train_config)})
        # Keep the model with the highest user defined score.
    
        filename = f"{{epoch}}-{{{train_config.optimize}:.2f}}"
        checkpoint_callback = ModelCheckpoint(dirpath=Path(train_config.root_dir) / train_config.model_checkpoint_dir, 
                                              filename=filename, monitor=train_config.optimize, 
                                              mode=train_config.optimize_mode, verbose=True, save_top_k=1)
        early_callback = EarlyStopping(monitor=train_config.optimize, min_delta=train_config.min_delta, 
                                       patience=train_config.patience, verbose=True, mode=train_config.optimize_mode)
        # Run the training loop.
        trainer = Trainer(callbacks=[checkpoint_callback, early_callback], default_root_dir=train_config.root_dir,
                          fast_dev_run=bool(train_config.debug_mode_sample), max_epochs=train_config.max_epochs, 
                          max_time=train_config.max_time, precision=train_config.precision,
                          logger=[mlf_logger, csv_logger], accumulate_grad_batches=train_config.accumulate_grad_batches, 
                          deterministic=train_config.deterministic,**lightning_trainer_args)
        
        trainer.fit(model=light_mod, datamodule=data_module)
        best_model_path = checkpoint_callback.best_model_path

        # Evaluate the last and the best models on the test sample.
        trainer.test(model=light_mod, datamodule=data_module)
        trainer.test(model=light_mod, datamodule=data_module, ckpt_path=best_model_path)
        
        if use_best_model:
            light_mod = TransformerModule.load_from_checkpoint(best_model_path, model=model)
            light_mod.model.save_pretrained(Path(train_config.root_dir) / train_config.adapter_output) # it only saves PEFT adapters
            
    return light_mod, data_module, best_model_path


def read_labels(self, label: str | pd.Series) -> str | pd.Series:
    """
    Reads the label data from a file or returns the input data.

    Parameters
    ----------
    label : str or pd.Series
        The label data.

    Returns
    -------
    pd.Series
        The label data as a pandas Series.
    """
    match label:
        case pd.Series() as labels:
            return labels.to_numpy()
        
        case pd.DataFrame() as labels:
            return labels.squeeze().to_numpy()
        
        case str() | Path() as labels if Path(labels).exists() and Path(labels).suffix == ".csv":
            labels = pd.read_csv(labels, index_col=0)
            return labels.squeeze().to_numpy()
        
        case str() | Path() as labels if Path(labels).exists() and Path(labels).suffix in [".npy", ".npz"]:
            if Path(labels).suffix == ".npz":
                return list(np.load(labels).values())[0]
            return np.load(labels)
        
        case list() | np.ndarray() as labels:
            return np.array(labels)
        case _:
            raise ValueError(f"label should be a csv file, an array, a pandas Series, DataFrame: you provided {label}")
        
        
def main():
    args = parse_args()
    # Convert JSON strings to dictionaries
    tokenizer_args = load_config(args.tokenizer_args, extension=args.tokenizer_args.split(".")[-1])
    lightning_trainer_args = load_config(args.lightning_trainer_args, extension=args.lightning_trainer_args.split(".")[-1])
    if args.lora_init == "True":
        lora_init = True
    elif args.lora_init == "False":
        lora_init = False
    else:
        lora_init = args.lora_init
    
    # Load label array from file
    label = read_labels(args.label)
    # Placeholder for loading configurations from files if provided
    train_config = TrainConfig(**load_config(args.train_config, extension=args.train_config.split(".")[-1]))
    llm_config = LLMConfig(**load_config(args.llm_config, extension=args.llm_config.split(".")[-1]))
    split_config = SplitConfig(**load_config(args.split_config, extension=args.split_config.split(".")[-1]))

    # Placeholder for the training loop call
    model, data_module, best_model_path = training_loop(args.fasta_file, label, args.lr,
                                                        train_config, llm_config, split_config,
                                                        tokenizer_args, lightning_trainer_args,
                                                        args.use_best_model, lora_init)

if __name__ == "__main__":
    main()