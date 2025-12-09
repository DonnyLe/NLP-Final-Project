# plm_utils.py

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from transcript_preprocessing import get_stratified_kfold_splits

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================
# Metrics
# =========================

def compute_metrics_from_labels(labels, preds) -> Dict[str, float]:
    """
    Compute accuracy, macro/weighted F1, precision, recall from labels and preds.
    """
    acc = accuracy_score(labels, preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
    }


def compute_metrics_for_logits(eval_pred):
    """
    HuggingFace Trainer metrics hook.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics_from_labels(labels, preds)


# =========================
# Tokenization
# =========================

def tokenize_batch_factory(transcript_col: str, tokenizer, max_len: int):
    def tokenize_batch(batch):
        return tokenizer(
            batch[transcript_col],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
    return tokenize_batch


# =========================
# Upsampling + class weights
# =========================

def upsample_train_df(
    train_df: pd.DataFrame,
    label_col: str = "Label",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simple minority upsampling to match the largest class size.
    """
    rng = np.random.default_rng(seed)

    labels, counts = np.unique(train_df[label_col].values, return_counts=True)
    max_count = counts.max()

    upsampled_parts = []

    for label in labels:
        class_df = train_df[train_df[label_col] == label]
        n_current = len(class_df)

        if n_current < max_count:
            extra_indices = rng.choice(
                class_df.index.values,
                size=max_count - n_current,
                replace=True,
            )
            extra_df = train_df.loc[extra_indices]
            balanced_df = pd.concat([class_df, extra_df], axis=0)
        else:
            balanced_df = class_df

        upsampled_parts.append(balanced_df)

    upsampled_df = pd.concat(upsampled_parts, axis=0)
    upsampled_df = upsampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return upsampled_df


def compute_class_weights_from_df(
    train_df: pd.DataFrame,
    label_col: str = "Label",
) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.
    weight_c = N / (num_classes * count_c)
    """
    labels = train_df[label_col].values
    class_indices, counts = np.unique(labels, return_counts=True)

    num_classes = len(class_indices)
    total = counts.sum()

    weights = total / (num_classes * counts.astype(np.float32))

    sorted_weights = np.zeros(num_classes, dtype=np.float32)
    for cls, w in zip(class_indices, weights):
        sorted_weights[int(cls)] = w

    return torch.tensor(sorted_weights, dtype=torch.float32)


class WeightedCETrainer(Trainer):
    """
    Trainer subclass with class-weighted CrossEntropyLoss.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self._loss_fct = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self._loss_fct is None:
            self._loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )

        loss = self._loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


# =========================
# Hyperparameter sampling
# =========================

def sample_hyperparams(model_name: str, rng: np.random.Generator) -> Dict:
    """
    Sample one random hyperparameter configuration.

    Uses a tighter space for big models (e.g., roberta-large / bert-large).
    """
    name_lower = model_name.lower()

    if "roberta-large" in name_lower or "bert-large" in name_lower or "ernie" in name_lower:
        # Large model: more conservative + smaller batches
        learning_rate = float(rng.choice([1e-5, 1.5e-5, 2e-5]))
        batch_size = int(rng.choice([2, 4]))
        num_train_epochs = int(rng.choice([3, 4]))
        max_len = int(rng.choice([256, 384]))
        weight_decay = float(rng.uniform(0.0, 0.05))
        warmup_ratio = float(rng.choice([0.0, 0.1]))
    else:
        # Base model: slightly wider space
        learning_rate = float(rng.choice([1e-5, 2e-5, 3e-5, 5e-5]))
        batch_size = int(rng.choice([8, 16]))
        num_train_epochs = int(rng.choice([3, 4, 5]))
        max_len = int(rng.choice([256, 384, 512]))
        weight_decay = float(rng.uniform(0.0, 0.1))
        warmup_ratio = float(rng.choice([0.0, 0.1, 0.2]))

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_train_epochs": num_train_epochs,
        "max_len": max_len,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
    }


# =========================
# Core training helpers
# =========================

def _build_tokenized_datasets(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    transcript_col: str,
    label_col: str,
    tokenizer,
    max_len: int,
) -> Tuple[Dataset, Dataset]:
    tokenize_batch = tokenize_batch_factory(transcript_col, tokenizer, max_len)

    train_ds = Dataset.from_pandas(train_df)
    dev_ds = Dataset.from_pandas(dev_df)

    train_tok = train_ds.map(tokenize_batch, batched=True)
    dev_tok = dev_ds.map(tokenize_batch, batched=True)

    train_tok = train_tok.rename_column(label_col, "labels")
    dev_tok = dev_tok.rename_column(label_col, "labels")

    cols_to_remove = [
        "Record-ID",
        "Class",
        "Transcript_PFT",
        "Transcript_CTD",
        "Transcript_SFT",
        "Transcript_ALL",
        "__index_level_0__",
        "Label_3cls",
    ]
    cols_to_remove = [c for c in cols_to_remove if c in train_tok.column_names]
    train_tok = train_tok.remove_columns(cols_to_remove)
    dev_tok = dev_tok.remove_columns(cols_to_remove)

    train_tok.set_format("torch")
    dev_tok.set_format("torch")

    return train_tok, dev_tok


def train_eval_one_fold(
    df: pd.DataFrame,
    transcript_col: str,
    label_col: str,
    tokenizer,
    model_name: str,
    max_len: int,
    learning_rate: float,
    num_train_epochs: int,
    batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    fold_idx: int,
    trial_id: int,
    seed: int,
    use_upsampling: bool = False,
    use_class_weights: bool = False,
    tmp_dir_tag: str = "tmp",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One fold of training + eval (for either binary or multi-class).
    """
    if use_upsampling and use_class_weights:
        raise ValueError("Choose either upsampling OR class-weighted loss, not both.")

    suffix = ""
    if use_upsampling:
        suffix = " (upsampling)"
    elif use_class_weights:
        suffix = " (class-weighted)"

    print(f"  Fold {fold_idx}, trial {trial_id}{suffix}")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    dev_df = df.iloc[dev_idx].reset_index(drop=True)

    if use_upsampling:
        train_df = upsample_train_df(train_df, label_col=label_col, seed=seed + fold_idx)

    if use_class_weights:
        class_weights = compute_class_weights_from_df(train_df, label_col=label_col)
    else:
        class_weights = None

    train_tok, dev_tok = _build_tokenized_datasets(
        train_df=train_df,
        dev_df=dev_df,
        transcript_col=transcript_col,
        label_col=label_col,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    num_labels = df[label_col].nunique()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    output_dir = f"{tmp_dir_tag}_fold_{fold_idx}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=False,
        logging_steps=50,
        save_total_limit=1,
        dataloader_num_workers=0,
        report_to=[],
    )

    if use_class_weights and class_weights is not None:
        trainer = WeightedCETrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=dev_tok,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_for_logits,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=dev_tok,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_for_logits,
        )

    trainer.train()
    _ = trainer.evaluate()

    preds_output = trainer.predict(dev_tok)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = np.array(dev_tok["labels"])

    return y_true, y_pred


def run_kfold_trial(
    df: pd.DataFrame,
    transcript_col: str,
    label_col: str,
    model_name: str,
    max_len: int,
    learning_rate: float,
    num_train_epochs: int,
    batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    n_splits: int,
    seed: int,
    trial_id: int,
    use_upsampling: bool = False,
    use_class_weights: bool = False,
    tmp_dir_tag: str = "tmp",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Full stratified K-fold CV for a single hyperparameter configuration.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []

    folds = list(
        get_stratified_kfold_splits(
            df,
            transcript_col=transcript_col,
            label_col=label_col,
            n_splits=n_splits,
            seed=seed,
        )
    )

    for fold_idx, train_idx, dev_idx in folds:
        y_true_fold, y_pred_fold = train_eval_one_fold(
            df=df,
            transcript_col=transcript_col,
            label_col=label_col,
            tokenizer=tokenizer,
            model_name=model_name,
            max_len=max_len,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            train_idx=train_idx,
            dev_idx=dev_idx,
            fold_idx=fold_idx,
            trial_id=trial_id,
            seed=seed,
            use_upsampling=use_upsampling,
            use_class_weights=use_class_weights,
            tmp_dir_tag=tmp_dir_tag,
        )
        all_y_true.append(y_true_fold)
        all_y_pred.append(y_pred_fold)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    metrics = compute_metrics_from_labels(y_true_all, y_pred_all)
    return metrics, y_true_all, y_pred_all


def random_search_kfold(
    df: pd.DataFrame,
    transcript_col: str,
    label_col: str,
    model_name: str,
    model_tag: str,
    n_trials: int,
    n_splits: int,
    seed: int,
    results_csv_path: str,
    use_upsampling: bool = False,
    use_class_weights: bool = True,
) -> Tuple[pd.DataFrame, Dict, np.ndarray, np.ndarray]:
    """
    Generic random search wrapper with full K-fold CV.
    """
    if use_upsampling and use_class_weights:
        raise ValueError("Choose either upsampling OR class-weighted loss, not both.")

    rng = np.random.default_rng(seed)

    all_results: List[Dict] = []
    best_macro_f1 = -1.0
    best_cfg: Dict = {}
    best_y_true: Optional[np.ndarray] = None
    best_y_pred: Optional[np.ndarray] = None

    mode = "baseline"
    if use_upsampling:
        mode = "upsampling"
    elif use_class_weights:
        mode = "class-weighted"

    print(
        f"\nRandom search ({mode}) for {transcript_col}, label_col={label_col}, "
        f"{n_trials} trials, {n_splits}-fold CV, model={model_name}"
    )

    for trial in range(1, n_trials + 1):
        print(f"\nTrial {trial}/{n_trials}")
        cfg = sample_hyperparams(model_name, rng)
        for k, v in cfg.items():
            print(f"  {k}: {v}")

        metrics, y_true_all, y_pred_all = run_kfold_trial(
            df=df,
            transcript_col=transcript_col,
            label_col=label_col,
            model_name=model_name,
            max_len=cfg["max_len"],
            learning_rate=cfg["learning_rate"],
            num_train_epochs=cfg["num_train_epochs"],
            batch_size=cfg["batch_size"],
            weight_decay=cfg["weight_decay"],
            warmup_ratio=cfg["warmup_ratio"],
            n_splits=n_splits,
            seed=seed,
            trial_id=trial,
            use_upsampling=use_upsampling,
            use_class_weights=use_class_weights,
            tmp_dir_tag=f"{model_tag}_rs",
        )

        trial_result = {
            "trial_id": trial,
            **cfg,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
        }
        all_results.append(trial_result)

        print("  CV accuracy:", metrics["accuracy"])
        print("  CV macro_f1:", metrics["macro_f1"])
        print("  CV weighted_f1:", metrics["weighted_f1"])

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_cfg = {**trial_result}
            best_y_true = y_true_all
            best_y_pred = y_pred_all
            print("  New best configuration for this task.")

        pd.DataFrame(all_results).to_csv(results_csv_path, index=False)

    results_df = pd.DataFrame(all_results).sort_values("macro_f1", ascending=False)
    return results_df, best_cfg, best_y_true, best_y_pred


# =========================
# Single-split trainer (used in cascade)
# =========================

def train_binary_single_split(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    transcript_col: str,
    label_col: str,
    model_name: str,
    max_len: int,
    learning_rate: float,
    num_train_epochs: int,
    batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    use_class_weights: bool = True,
    use_upsampling: bool = False,
    seed: int = 42,
    tmp_dir_tag: str = "tmp_single",
) -> np.ndarray:
    """
    Train on train_df, predict labels on dev_df.
    Returns predicted labels (numpy array) for dev_df in order.
    """
    if use_upsampling and use_class_weights:
        raise ValueError("Choose either upsampling OR class-weighted loss, not both.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df_local = train_df.reset_index(drop=True)
    dev_df_local = dev_df.reset_index(drop=True)

    if use_upsampling:
        train_df_local = upsample_train_df(
            train_df_local, label_col=label_col, seed=seed
        )

    if use_class_weights:
        class_weights = compute_class_weights_from_df(
            train_df_local, label_col=label_col
        )
    else:
        class_weights = None

    train_tok, dev_tok = _build_tokenized_datasets(
        train_df=train_df_local,
        dev_df=dev_df_local,
        transcript_col=transcript_col,
        label_col=label_col,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    num_labels = train_df_local[label_col].nunique()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=f"{tmp_dir_tag}",
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=False,
        logging_steps=100,
        save_total_limit=1,
        dataloader_num_workers=0,
        report_to=[],
    )

    if use_class_weights and class_weights is not None:
        trainer = WeightedCETrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=None,
            tokenizer=tokenizer,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=None,
            tokenizer=tokenizer,
        )

    trainer.train()

    preds_output = trainer.predict(dev_tok)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    return y_pred
