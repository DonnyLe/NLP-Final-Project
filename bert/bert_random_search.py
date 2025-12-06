import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transcript_preprocessing import get_stratified_kfold_splits, load_transcript_splits
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

CLASS_NAMES = ["HC", "MCI", "Dementia"]

# Avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------
# Metric + tokenization helpers
# -------------------------

def tokenize_batch_factory(transcript_col: str, tokenizer, max_len: int):
    def tokenize_batch(batch):
        return tokenizer(
            batch[transcript_col],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
    return tokenize_batch


def compute_metrics_for_logits(eval_pred):
    """
    For Trainer: compute accuracy, macro/weighted F1, precision, recall from logits.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics_from_labels(labels, preds)


def compute_metrics_from_labels(labels, preds) -> Dict[str, float]:
    """
    Compute accuracy, macro/weighted F1, precision, and recall from labels and preds.
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


# -------------------------
# Class weights helper
# -------------------------

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

    # Ensure weights are ordered by class index 0..num_classes-1
    sorted_weights = np.zeros(num_classes, dtype=np.float32)
    for cls, w in zip(class_indices, weights):
        sorted_weights[int(cls)] = w

    return torch.tensor(sorted_weights, dtype=torch.float32)


# -------------------------
# Custom Trainer with weighted loss
# -------------------------

class WeightedCETrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self._loss_fct = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Initialize loss function on first call
        if self._loss_fct is None:
            self._loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )

        loss = self._loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


# -------------------------
# Train/eval on a single fold
# -------------------------

def train_eval_one_fold(
    df: pd.DataFrame,
    transcript_col: str,
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train on one fold with class-weighted loss.
    Returns labels and predictions for that fold's dev split.
    """
    print(f"  Fold {fold_idx}: training (trial {trial_id})")

    tokenize_batch = tokenize_batch_factory(transcript_col, tokenizer, max_len)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    dev_df = df.iloc[dev_idx].reset_index(drop=True)

    # Compute class weights from this fold's training split
    class_weights = compute_class_weights_from_df(train_df, label_col="Label")

    train_ds = Dataset.from_pandas(train_df)
    dev_ds = Dataset.from_pandas(dev_df)

    train_tok = train_ds.map(tokenize_batch, batched=True)
    dev_tok = dev_ds.map(tokenize_batch, batched=True)

    # Rename Label -> labels for Trainer
    train_tok = train_tok.rename_column("Label", "labels")
    dev_tok = dev_tok.rename_column("Label", "labels")

    # Drop unused columns if present
    cols_to_remove = [
        "Record-ID",
        "Class",
        "Transcript_PFT",
        "Transcript_CTD",
        "Transcript_SFT",
        "Transcript_ALL",
        "__index_level_0__",
    ]
    cols_to_remove = [c for c in cols_to_remove if c in train_tok.column_names]
    train_tok = train_tok.remove_columns(cols_to_remove)
    dev_tok = dev_tok.remove_columns(cols_to_remove)

    train_tok.set_format("torch")
    dev_tok.set_format("torch")

    num_labels = df["Label"].nunique()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # Trainer needs an output directory but we don't care about checkpoints
    output_dir = "bert_tmp"
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

    trainer = WeightedCETrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_for_logits,
    )

    trainer.train()
    _ = trainer.evaluate()

    # Predictions on this fold's dev set
    preds_output = trainer.predict(dev_tok)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = np.array(dev_tok["labels"])

    return y_true, y_pred


# -------------------------
# Full k-fold CV for one trial
# -------------------------

def run_kfold_bert_trial(
    df: pd.DataFrame,
    transcript_col: str,
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
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Run full stratified K-fold CV for one hyperparameter trial.
    Returns aggregated metrics over all folds plus concatenated labels/preds.
    """
    print(
        f"Running full {n_splits}-fold CV for {transcript_col}, "
        f"trial {trial_id}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []

    folds = list(
        get_stratified_kfold_splits(
            df,
            transcript_col=transcript_col,
            label_col="Label",
            n_splits=n_splits,
            seed=seed,
        )
    )

    for fold_idx, train_idx, dev_idx in folds:
        y_true_fold, y_pred_fold = train_eval_one_fold(
            df=df,
            transcript_col=transcript_col,
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
        )
        all_y_true.append(y_true_fold)
        all_y_pred.append(y_pred_fold)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    metrics = compute_metrics_from_labels(y_true_all, y_pred_all)

    return metrics, y_true_all, y_pred_all


# -------------------------
# Random hyperparameter sampler
# -------------------------

def sample_hyperparams(rng: np.random.Generator) -> Dict:
    """
    Sample one random hyperparameter configuration.
    """
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


# -------------------------
# Random search with full k-fold CV
# -------------------------

def random_search_kfold(
    df: pd.DataFrame,
    transcript_col: str,
    model_name: str = "bert-base-uncased",
    n_trials: int = 20,
    n_splits: int = 5,
    seed: int = 42,
    results_csv_path: str = "bert_random_search_results.csv",
) -> Tuple[pd.DataFrame, Dict, np.ndarray, np.ndarray]:
    """
    Run random search over hyperparameters using full K-fold CV for each trial.
    Uses class-weighted loss (no upsampling).
    """
    rng = np.random.default_rng(seed)

    all_results: List[Dict] = []
    best_macro_f1 = -1.0
    best_cfg: Dict = {}
    best_y_true: Optional[np.ndarray] = None
    best_y_pred: Optional[np.ndarray] = None

    print(
        f"\nRandom search for {transcript_col} with {n_trials} trials and "
        f"{n_splits} fold CV (class-weighted loss)"
    )

    for trial in range(1, n_trials + 1):
        print(f"\nTrial {trial}/{n_trials} for {transcript_col}: ")

        cfg = sample_hyperparams(rng)
        print("Hyperparameters:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")

        metrics, y_true_all, y_pred_all = run_kfold_bert_trial(
            df=df,
            transcript_col=transcript_col,
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

        print("Trial CV metrics:")
        print(f"  accuracy = {metrics['accuracy']}")
        print(f"  macro_f1 = {metrics['macro_f1']}")
        print(f"  weighted_f1 = {metrics['weighted_f1']}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_cfg = {**trial_result}
            best_y_true = y_true_all
            best_y_pred = y_pred_all
            print("New best configuration so far for this transcript type.")

        # Save intermediate results after each trial
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_csv_path, index=False)

    results_df = pd.DataFrame(all_results).sort_values("macro_f1", ascending=False)
    return results_df, best_cfg, best_y_true, best_y_pred


# -------------------------
# Main: run on ALL transcript types
# -------------------------

if __name__ == "__main__":
    TRANSCRIPTS_CSV = "data/transcripts_cleaned.csv"

    # Number of trials and folds
    N_TRIALS = 15
    N_SPLITS = 5

    # Load raw data once so we can build Transcript_ALL
    raw_df = pd.read_csv(TRANSCRIPTS_CSV)

    # Build combined transcript
    raw_df["Transcript_ALL"] = (
        "[PFT] " + raw_df["Transcript_PFT"].fillna("") + " "
        + "[CTD] " + raw_df["Transcript_CTD"].fillna("") + " "
        + "[SFT] " + raw_df["Transcript_SFT"].fillna("")
    )

    # Save a temp CSV with the new column so load_transcript_splits can see it
    tmp_csv = "data/transcripts_with_all.csv"
    raw_df.to_csv(tmp_csv, index=False)

    TRANSCRIPT_COLS = [
        "Transcript_PFT",
        "Transcript_CTD",
        "Transcript_SFT",
        "Transcript_ALL",
    ]

    # This helper builds one df per transcript type, dropping rows with NaNs for that column
    df_by_transcript = load_transcript_splits(
        tmp_csv,
        transcript_cols=TRANSCRIPT_COLS,
    )

    best_overall = []

    for transcript_col, df in df_by_transcript.items():
        print(f"\nStarting random search for {transcript_col} (class-weighted loss)")

        results_csv_path = f"bert_random_search_{transcript_col}_folds_weighted.csv"

        results_df, best_cfg, y_true_best, y_pred_best = random_search_kfold(
            df=df,
            transcript_col=transcript_col,
            model_name="bert-base-uncased",
            n_trials=N_TRIALS,
            n_splits=N_SPLITS,
            seed=42,
            results_csv_path=results_csv_path,
        )

        best_row = best_cfg.copy()
        best_row["transcript_col"] = transcript_col
        best_overall.append(best_row)

        print(f"\nTop configs for {transcript_col}:")
        print(results_df.head())

        # Confusion matrix for best trial of this transcript type
        cm = confusion_matrix(y_true_best, y_pred_best, labels=[0, 1, 2])

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix - {transcript_col} (class-weighted)")

        cm_filename = f"bert_confusion_matrix_{transcript_col}_weighted.png"
        fig.savefig(cm_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        report = classification_report(
            y_true_best,
            y_pred_best,
            target_names=CLASS_NAMES,
            zero_division=0,
        )

        print(f"\nClassification report for {transcript_col} (class-weighted):")
        print(report)

        report_filename = f"bert_classification_report_{transcript_col}_weighted.txt"
        with open(report_filename, "w") as f:
            f.write(f"Classification report for {transcript_col} (class-weighted)\n\n")
            f.write(report)
            f.write("\n")
            f.write("\nBest trial CV metrics:\n")
            f.write(f"accuracy = {best_cfg['accuracy']}\n")
            f.write(f"macro_f1 = {best_cfg['macro_f1']}\n")
            f.write(f"weighted_f1 = {best_cfg['weighted_f1']}\n")

    best_df = pd.DataFrame(best_overall).sort_values("macro_f1", ascending=False)
    best_df.to_csv("bert_random_search_best_per_transcript_weighted.csv", index=False)

    summary_path = "bert_random_search_summary_weighted.txt"
    with open(summary_path, "w") as f:
        f.write("Best config per transcript type (sorted by macro_f1):\n\n")
        f.write(best_df.to_string(index=False))
        f.write("\n")

    print("\nFinished random search for all transcript types (class-weighted loss).")
    print("Best configs table:")
    print(best_df)
