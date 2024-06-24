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
from safetensors import SafetensorError
from functools import partial
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from typing import Iterable
from torchmetrics.classification import (
    Accuracy, F1Score, Precision, Recall, AUROC, AveragePrecision, CohenKappa, MatthewsCorrCoef) 
from torchmetrics.regression import (
    MeanAbsoluteError, MeanSquaredError,  PearsonCorrCoef, KendallRankCorrCoef, R2Score,
    MeanAbsolutePercentageError, MeanSquaredLogError)
import pandas as pd
import numpy as np
from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from .embeddings import TokenizeFasta
from .train_config import LLMConfig, SplitConfig, TrainConfig


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


def classification_metrics(split: str, num_classes: int=2, threshold: float=0.5):
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


def regression_metrics(split: str):
    
    metrics = {f"{split}_MAE": MeanAbsoluteError(),
                f"{split}_MSE": MeanSquaredError(squared=False),
                f"{split}_RMSE": MeanSquaredError(squared=True),
                f"{split}_R2": R2Score(),
                f"{split}_Pearson": PearsonCorrCoef(),
                f"{split}_Kendall": KendallRankCorrCoef(),
                f"{split}_MAPE": MeanAbsolutePercentageError(),
                f"{split}_MSLE": MeanSquaredLogError()}
    return metrics


@dataclass(slots=True)
class PreparePEFT:
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

    def setup_model(self):
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
    
    def prepare_model(self):
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
    cluster_file: str | None = None
    shuffle: bool = True
    random_seed: int = 21321412
    splitting_strategy: str = "random"
    num_split: int = 5
    stratify: bool | None = None
    test_size: float = 0.2
    
    def get_split_indices(self, dataset: Dataset):
        if self.splitting_strategy == "cluster":
            cluster = split.ClusterSpliter(self.cluster_file, self.num_split, 
                                           shuffle=self.shuffle, random_state=self.random_seed)
            train, test = cluster.train_test_split(range(len(dataset)), groups=dataset["id"])
        elif self.splitting_strategy == "random":
            stratify = dataset["labels"].cpu() if self.stratify else None
            train, test = train_test_split(range(len(dataset)), stratify=stratify, 
                                           test_size=self.test_size)
        return train, test

    def get_data(self, dataset: Dataset):
        train_indices, test_indices = self.get_split_indices(dataset)
        train_, test = dataset.select(train_indices), dataset.select(test_indices)
        train_indices, validation_indices = self.get_split_indices(train_)
        train, validation = train_.select(train_indices), train_.select(validation_indices)
        return train, validation, test


class DataModule(LightningDataModule):
    def __init__(self, splitter: PrepareSplit, fasta_file: str | Path, label: np.array, 
                 config: LLMConfig = LLMConfig(), batch_size: int=1, tokenizer_args: dict=dict()) -> None:
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
        train, validation, test = self.splitter.get_data(self.dataset)
        self.train = train
        self.validation = validation
        self.test = test
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class TransformerModule(LightningModule):
    def __init__(
        self,
        model: PeftModel,
        train_config: dataclass,
        lr: float
    ):
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
        label: list[int] | None = None,
    ):
        """Calc the loss by passing inputs to the model and comparing against ground
        truth labels. Here, all of the arguments of self.model comes from the
        SequenceClassification head from HuggingFace.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label,
        )

    def _compute_predictions(self, batch) -> tuple:
        """Helper method hosting the evaluation logic common to the <split>_step methods."""
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
        outputs, preds = self._compute_predictions(batch)
        for metric in self.train_metrics.values():
            metric.to(preds.device)
            metric.update(preds, batch["labels"].to(preds.device))
        self.log("loss", outputs["loss"], on_epoch=True, on_step=False)

        return outputs["loss"] # the automodel has its own loss function depending on the problem
    
    def on_training_epoch_end(self):
        # Log the accumulated training metrics
        for name, metric in self.train_metrics.items():
            self.log(f'{name}', metric.compute())
            metric.reset()
            
    def validation_step(self, batch, batch_idx) -> dict[str, Any]:
        outputs, preds = self._compute_predictions(batch)
        for metric in self.val_metrics.values():
            metric.to(preds.device)
            metric.update(preds, batch["labels"].to(preds.device))
        self.log("loss", outputs["loss"], on_epoch=True, on_step=False)
        return outputs["loss"]
    
    def on_validation_epoch_end(self):
        # Log the accumulated validation metrics
        for name, metric in self.val_metrics.items():
            self.log(f'{name}', metric.compute())
            metric.reset()

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        outputs, preds = self._compute_predictions(batch)
        for metric in self.test_metrics.values():
            metric.to(preds.device)
            metric.update(preds, batch["labels"].to(preds.device))
        self.log("loss", outputs["loss"], on_epoch=True, on_step=False)
        
        return outputs["loss"]
    
    def on_test_epoch_end(self):
        # Log the accumulated test metrics
        for name, metric in self.test_metrics.items():
            self.log(f'{name}', metric.compute())
            metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Predict the output of the model.
        
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
                  use_best_model: bool = True) -> tuple[TransformerModule, DataModule, str]:
    """Train and checkpoint the model with highest score; log that model to MLflow and
    return it."""
    seed_everything(split_config.random_seed, workers=True)
    splitter = PrepareSplit(split_config.cluster_file, split_config.shuffle, split_config.random_seed, 
                            split_config.splitting_strategy, 
                            split_config.num_split, split_config.stratify, split_config.test_size)
    
    data_module = DataModule(splitter, fasta_file, label, llm_config, train_config.batch_size, tokenizer_args)

    peft = PreparePEFT(train_config, llm_config)
    model = peft.prepare_model()
    light_mod = TransformerModule(model, train_config, lr=lr)

    # Set Up MLflow context manager.
    mlflow.set_experiment(train_config.mlflow_experiment_name)
    with mlflow.start_run(run_name=train_config.mlflow_run_name, description=train_config.mlflow_description) as run:
        # TODO: MLflow metrics should show epochs rather than steps on the x-axis
        
        mlf_logger = MLFlowLogger(experiment_name=mlflow.get_experiment(run.info.experiment_id).name, 
                                  tracking_uri=mlflow.get_tracking_uri(), log_model=True)
        
        mlf_logger._run_id = run.info.run_id
        mlflow.log_params({k: v for k, v in asdict(train_config).items() if not k.startswith("mlflow_")})
        mlflow.log_params({k: v for k, v in asdict(split_config).items() if not k.startswith("mlflow_")})
        mlflow.log_params({k: v for k, v in asdict(llm_config).items() if not k not in asdict(train_config)})
        # Keep the model with the highest user defined score.
    
        filename = f"{{epoch}}-{{{train_config.optimize}:.2f}}"
        checkpoint_callback = ModelCheckpoint(filename=filename, monitor=train_config.optimize, 
                                              mode=train_config.optimize_mode, verbose=True, save_top_k=1)
        early_callback = EarlyStopping(monitor=train_config.optimize, min_delta=train_config.min_delta, 
                                       patience=train_config.patience, verbose=True, mode=train_config.optimize_mode)
        # Run the training loop.
        trainer = Trainer(callbacks=[checkpoint_callback, early_callback], default_root_dir=train_config.model_checkpoint_dir,
                          fast_dev_run=bool(train_config.debug_mode_sample), max_epochs=train_config.max_epochs, 
                          max_time=train_config.max_time, precision=train_config.precision,
                          logger=mlf_logger, accumulate_grad_batches=train_config.accumulate_grad_batches, 
                          deterministic=train_config.deterministic,**lightning_trainer_args)
        
        trainer.fit(model=light_mod, datamodule=data_module)
        best_model_path = checkpoint_callback.best_model_path

        # Evaluate the last and the best models on the test sample.
        trainer.test(model=light_mod, datamodule=data_module)
        trainer.test(model=light_mod, datamodule=data_module, ckpt_path=best_model_path)
        
        if use_best_model:
            light_mod = TransformerModule.load_from_checkpoint(best_model_path, model=model)
            light_mod.model.save_pretrained(train_config.adapter_output) # it only saves PEFT adapters
            
    return light_mod, data_module, best_model_path

