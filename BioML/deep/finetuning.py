from transformers import AutoModelForSequenceClassification, PreTrainedModel, BitsAndBytesConfig
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import argparse
from typing import Iterable, Any
from sklearn.model_selection import train_test_split
from BioML.utilities import split_methods as split
from lightning import LightningModule, LightningDataModule, Trainer
import lightning as L
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from dataclasses import dataclass, asdict
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from typing import Iterable
from torchmetrics.functional.classification import (
    accuracy, f1_score, precision, recall, auroc, average_precision, cohen_kappa, confusion_matrix, 
    matthews_corrcoef) 
from torchmetrics.functional.regression import (
    mean_absolute_error, mean_squared_error,  pearson_corrcoef, kendall_rank_corrcoef, r2_score,
    mean_absolute_percentage_error, mean_squared_log_error)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from ..utilities.utils import set_seed
from .embeddings import TokenizeFasta
from .train_config import LLMConfig, SplitConfig, TrainConfig
from ..models.metrics import ndcg_at_k


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


def classification_metrics(split: str, loss: torch.tensor, preds: torch.tensor, 
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


def regression_metrics(split: str, loss: torch.tensor, preds: torch.tensor, 
                                 target: torch.tensor):
    metrics = {f"{split}_Loss": loss,
                f"{split}_MAE": mean_absolute_error(preds, target),
                f"{split}_MSE": mean_squared_error(preds, target),
                f"{split}_RMSE": mean_squared_error(preds, target, squared=False),
                f"{split}_R2": r2_score(preds, target),
                f"{split}_Pearson": pearson_corrcoef(preds, target),
                f"{split}_Kendall": kendall_rank_corrcoef(preds, target),
                f"{split}_MAPE": mean_absolute_percentage_error(preds, target),
                f"{split}_MSLE": mean_squared_log_error(preds, target),
                f"{split}_NDCG": ndcg_at_k(target, preds, k=10, penalty=15)}
    return metrics


@dataclass(slots=True)
class PreparePEFT:
    qlora: bool = False
    gradient_cheeckpointing: bool = False
    
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
                        lora_dropout: float=0.05):
        
        if lora_alpha is None:
            lora_alpha = rank * 2
        else:
            print("Warning lora_alpha is set to a value. For optimal performance, it is recommended to set it double the rank")
        
        # get the lora models
        peft_config = LoraConfig(inference_mode=False, r=rank, lora_alpha=lora_alpha, 
                                 lora_dropout=lora_dropout, 
                                 target_modules=target_modules)
        
        return peft_config

    def get_model(self, model_params: LLMConfig):
        if self.qlora:
            bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            bnb_config = None
        
        device = "auto" if model_params.device == "cuda" or self.qlora else model_params.device
        model = AutoModelForSequenceClassification.from_pretrained(model_params.model_name, 
                                                                num_labels=model_params.num_classes, 
                                                                low_cpu_mem_usage=True,
                                                                torch_dtype= model_params.dtype, 
                                                                quantization_config=bnb_config,
                                                                device_map=device)
        if self.qlora:
            model = prepare_model_for_kbit_training(model)
        return model

@dataclass
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
            train, test = cluster.train_test_split(range(len(dataset)), groups=dataset["id"])
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


class DataModule(LightningDataModule):
    def __init__(self, splitter: PrepareSplit, fasta_file: str, config: LLMConfig,
                 batch_size: int=1, tokenizer_args: dict=dict()) -> None:
        super().__init__()
        self.splitter = splitter
        self.fasta_file = fasta_file
        self.batch_size = batch_size
        self.config = config
        self.tokenizer_args = tokenizer_args 
    
    def prepare_data(self) -> None:
        """Tokenize the fasta file and store the dataset in the class instance."""
        tokenizer = TokenizeFasta(self.config, tokenizer_args=self.tokenizer_args)
        self.dataset = tokenizer.tokenize(self.fasta_file)
        
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
        model: PreTrainedModel,
        peft_config: LoraConfig,
        objective: str,
        lr: float
    ):
        super().__init__()

        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        self.objective = objective
        self.metrics = {"classifcation": classification_metrics, 
                        "regression": regression_metrics}[self.objective]
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
        scheduler = OneCycleLR(optim, max_lr=self.lr, total_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": optim, "lr_scheduler": scheduler}


def finetune(cluster_file, shuffle, random_seed, splitting_strategy, num_split, stratify, test_size, fasta_file, config, 
             batch_size, tokenizer_args, qlora, train_config, *args):
    splitter = PrepareSplit(cluster_file, shuffle, random_seed, splitting_strategy, num_split, stratify, test_size)
    data_module = DataModule(splitter, fasta_file, config, batch_size, tokenizer_args)

    peft = PreparePEFT(qlora)
    model = peft.get_model(config)
    peft_config = peft.get_lora_config(rank=64, 
                                       target_modules=["key", "query", "value", "attention.dense.output"], 
                                       lora_dropout=0.05)
    
    model = TransformerModule(model, peft_config, train_config.objective, lr=1e-5)
    trainer = Trainer(max_epochs=train_config.max_epochs)
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())

def training_loop(llm_config: dataclass, train_config: TrainConfig, split_config: SplitConfig, 
                  tokenizer_args: dict=dict()) -> TransformerModule:
    """Train and checkpoint the model with highest F1; log that model to MLflow and
    return it."""
    splitter = PrepareSplit(split_config.cluster_file, split_config.shuffle, split_config.random_seed, 
                            split_config.splitting_strategy, 
                            split_config.num_split, split_config.stratify, split_config.test_size)
    
    data_module = DataModule(splitter, train_config.fasta_file, llm_config, train_config.batch_size, tokenizer_args)

    peft = PreparePEFT(train_config.qlora)
    model = peft.get_model(llm_config)
    peft_config = peft.get_lora_config(rank=64, target_modules=["key", "query", "value", "attention.dense.output"], 
                                       lora_dropout=0.05)
    
    model = TransformerModule(model, peft_config, train_config.objective, lr=1e-5)

    # Wire up MLflow context manager to Azure ML.
    mlflow.set_experiment(train_config.mlflow_experiment_name)

    with mlflow.start_run(run_name=train_config.mlflow_run_name, description=train_config.mlflow_description) as run:
        # Connect Lightning's MLFlowLogger plugin to azureml-mlflow as defined in the
        # context manager. TODO: MLflow metrics should show epochs rather than steps on
        #  the x-axis
        mlf_logger = MLFlowLogger(experiment_name=mlflow.get_experiment(run.info.experiment_id).name, 
                                  tracking_uri=mlflow.get_tracking_uri(), log_model=True)
        
        mlf_logger._run_id = run.info.run_id
        mlflow.log_params({k: v for k, v in asdict(train_config).items() if not k.startswith("mlflow_")})
        mlflow.log_params({k: v for k, v in asdict(split_config).items() if not k.startswith("mlflow_")})
        mlflow.log_params({k: v for k, v in asdict(llm_config).items() if not k not in asdict(train_config)})
        # Keep the model with the highest F1 score.
    
        filename = f"{{epoch}}-{train_config.optimize}{{:.2f}}"
        checkpoint_callback = ModelCheckpoint(filename=filename, monitor=train_config.optimize, mode=train_config.optimize_mode, 
                                              verbose=True, save_top_k=1)
        early_callback = EarlyStopping(monitor=train_config.optimize, min_delta=train_config.min_delta, 
                                       patience=train_config.patience, verbose=True, mode=train_config.optimize_mode)
        # Run the training loop.
        trainer = Trainer(callbacks=[checkpoint_callback, early_callback], default_root_dir=train_config.model_checkpoint_dir,
                          fast_dev_run=bool(train_config.debug_mode_sample), max_epochs=train_config.max_epochs, 
                          max_time=train_config.max_time, precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
                          logger=mlf_logger)
        
        trainer.fit(model=model, datamodule=data_module)
        best_model_path = checkpoint_callback.best_model_path

        # Evaluate the last and the best models on the test sample.
        trainer.test(model=model, datamodule=data_module)
        trainer.test(model=model, datamodule=data_module, ckpt_path=best_model_path)

    return model, data_module
