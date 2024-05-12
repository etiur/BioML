from transformers import AutoModelForSequenceClassification, PreTrainedModel
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import argparse
from typing import Iterable
from sklearn.model_selection import train_test_split
from BioML.utilities import split_methods as split
from lightning import LightningModule, LightningDataModule
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from dataclasses import dataclass
from peft import get_peft_model, LoraConfig
from typing import Iterable
from torchmetrics.functional.classification import (
    accuracy, f1_score, precision, recall, auroc, average_precision, cohen_kappa, confusion_matrix, 
    matthews_corrcoef) 
from torchmetrics.functional.regression import (
    mean_absolute_error, mean_squared_error,  pearson_corrcoef, kendall_rank_corrcoef, r2_score,
    mean_absolute_percentage_error, mean_squared_log_error)
from ..utilities.utils import set_seed
from embeddings import TokenizeFasta
from train_config import LLMConfig


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


def calculate_classification_metrics(split: str, loss: torch.tensor, preds: torch.tensor, 
                                     target: torch.tensor, num_classes: int=2, threshold: float=0.5):
    task = "binary" if num_classes == 2 else "multiclass"
    metrics = {f"{split}_Loss": loss,
                f"{split}_Acc": accuracy(preds=preds, target=target, num_classes=num_classes, task=task, 
                                         threshold=threshold, average="weighted"),
                f"{split}_F1":f1_score(preds=preds, target=target, task=task, num_classes=num_classes, 
                                       average="weighted"),
                f"{split}_Precision": precision(preds=preds, target=target, task=task, num_classes=num_classes, 
                                                average="weighted"),
                f"{split}_Recall": recall(preds=preds, target=target, task=task, num_classes=num_classes, 
                                          average="weighted"),
                f"{split}_MCC": matthews_corrcoef(preds=preds, target=target, num_classes=num_classes,
                                                  threshold=threshold, task=task),
                f"{split}_Confusion_Matrix": confusion_matrix(preds=preds, target=target, num_classes=num_classes, normalize="true", 
                                                              task=task, threshold=threshold),
                f"{split}_AUROC": auroc(preds=preds, target=target, num_classes=num_classes, task=task, 
                                        thresholds=None, average="weighted"),
                f"{split}_Average_Precision": average_precision(preds=preds, target=target, num_classes=num_classes, task=task, 
                                                                average="weighted"),
                f"{split}_Cohen_Kappa": cohen_kappa(preds=preds, target=target, num_classes=num_classes, 
                                                    task=task, threshold=threshold)}
    return metrics


def calculate_regression_metrics(split: str, loss: torch.tensor, preds: torch.tensor, 
                                 target: torch.tensor):
    metrics = {f"{split}_Loss": loss,
                f"{split}_MAE": mean_absolute_error(preds, target),
                f"{split}_MSE": mean_squared_error(preds, target),
                f"{split}_RMSE": mean_squared_error(preds, target, squared=False),
                f"{split}_R2": r2_score(preds, target),
                f"{split}_Pearson": pearson_corrcoef(preds, target),
                f"{split}_Kendall": kendall_rank_corrcoef(preds, target),
                f"{split}_MAPE": mean_absolute_percentage_error(preds, target),
                f"{split}_MSLE": mean_squared_log_error(preds, target)}
    return metrics


@dataclass(slots=True)
class PrepareSplit:
    cluster_file: str | None = None
    shuffle = True
    random_seed: int = 21321412
    splitting_strategy: str = "random"
    num_split: int = 5
    stratify: bool | None = None
    test_size: float = 0.2
    
    def get_split_indices(self, dataset: Dataset):
        if self.splitting_strategy == "cluster":
            cluster = split.ClusterSpliter(self.cluster_file, self.num_split, 
                                           shuffle=self.shuffle, random_state=self.random_seed)
            train, test = cluster.train_test_split(range(len(dataset)), index=dataset["id"])
        elif self.splitting_strategy == "random":
            stratify = dataset["labels"] if self.stratify else None
            train, test = train_test_split(range(len(dataset)), stratify=stratify, 
                                           test_size=self.test_size)
        return train, test

    def get_data(self, dataset: Dataset):
        train_indices, test_indices = self.get_split_indices(dataset)
        train_, test = dataset.select(train_indices), dataset.select(test_indices)
        train_indices, validation_indices = self.get_split_indices(train_)
        train, validation = train_.select(train_indices), train_.select(validation_indices)
        return train, validation, test


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


class DataModule(LightningDataModule):
    def __init__(self, splitter: PrepareSplit, dataset: Dataset,
                 batch_size: int=1) -> None:
        super().__init__()
        self.splitter = splitter
        self.dataset = dataset
        self.batch_size = batch_size
        
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
        model_params: LLMConfig,
        lr: float,
        peft_config: LoraConfig,
    ):
        super().__init__()

        model = AutoModelForSequenceClassification.from_pretrained(model_params.model_name, 
                                                                   num_labels=model_params.num_classes, 
                                                                   low_cpu_mem_usage=True)
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        self.objective = "classification" if model_params.num_classes >= 2 else "regression"
        self.metrics = {"classifcation": calculate_classification_metrics, 
                        "regression": calculate_regression_metrics}[self.objective]
        self.lr = lr

        self.save_hyperparameters()

    def forward(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        label: list[int],
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

    def _compute_metrics(self, batch, split) -> tuple:
        """Helper method hosting the evaluation logic common to the <split>_step methods."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            label=batch["label"],
        )

        # For predicting probabilities, do softmax along last dimension (by row).
        if self.objective == "classification":
            pred = torch.argmax(torch.softmax(outputs["logits"], dim=-1), dim=1)
        elif self.objective == "regression":
            pred = outputs["logits"]
        
        metrics = self.metrics(
            split=split,
            loss=outputs["loss"],
            preds=pred,
            target=batch["label"],
        )

        return outputs, metrics

    def training_step(self, batch, batch_idx):
        outputs, metrics = self._compute_metrics(batch, "Train")
        self.log_dict(metrics, on_epoch=True, on_step=False)

        return outputs["loss"]

    def validation_step(self, batch, batch_idx) -> dict[str, Any]:
        _, metrics = self._compute_metrics(batch, "Val")
        self.log_dict(metrics, on_epoch=True, on_step=False)

        return metrics

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        _, metrics = self._compute_metrics(batch, "Test")
        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self) -> Optimizer:
        optim = AdamW(
            params=self.parameters(),
            lr=self.lr)
        scheduler = OneCycleLR(optim, max_lr=self.lr, total_steps=self.trainer.estimated_stepping_batches,
                               )
        return {"optimizer": optim, "lr_scheduler": scheduler}