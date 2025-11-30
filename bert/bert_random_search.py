import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from transcript_preprocessing import get_stratified_kfold_splits, load_transcript_splits
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


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


def compute_metrics(eval_pred):
    """
    Compute a rich set of metrics for multi-class classification:
    - accuracy
    - macro_f1, weighted_f1
    - precision_macro, recall_macro
    - precision_weighted, recall_weighted
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)

    # Macro (all classes weighted equally)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    # Weighted (account for class imbalance)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
    }
    return metrics


# -------------------------
# Single-fold BERT training
# -------------------------

def run_single_fold_bert(
    df: pd.DataFrame,
    transcript_col: str,
    model_name: str,
    max_len: int,
    learning_rate: float,
    num_train_epochs: int,
    batch_size: int,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.0,
    n_splits: int = 5,
    seed: int = 42,
    fold_to_use: int = 1,  # 1-based, consistent with get_stratified_kfold_splits
    trial_id: int = 0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, str]:
    """
    Fine-tune BERT on a single stratified K-fold split for one transcript type.
    Uses 'Label' as the numeric label column (0=HC,1=MCI,2=Dementia).

    Returns:
        metrics_simple: dict with all key metrics + model_dir
        y_true: np.ndarray of true labels for the dev set
        y_pred: np.ndarray of predicted labels for the dev set
        model_dir: directory where this trial's model checkpoints are saved
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenize_batch = tokenize_batch_factory(transcript_col, tokenizer, max_len)

    # Pick the requested fold (1..n_splits)
    chosen_split = None
    for fold_idx, train_idx, dev_idx in get_stratified_kfold_splits(
        df,
        transcript_col=transcript_col,
        label_col="Label",   # IMPORTANT: numeric label column
        n_splits=n_splits,
        seed=seed,
    ):
        if fold_idx == fold_to_use:
            chosen_split = (fold_idx, train_idx, dev_idx)
            break

    if chosen_split is None:
        raise ValueError(
            f"Requested fold_to_use={fold_to_use}, but only "
            f"{n_splits} folds are available (1..{n_splits})."
        )

    fold_idx, train_idx, dev_idx = chosen_split
    print(f"\n=== Single-fold training: fold {fold_idx} / {n_splits} ({transcript_col}), "
          f"trial {trial_id} ===")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    dev_df   = df.iloc[dev_idx].reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df)
    dev_ds   = Dataset.from_pandas(dev_df)

    train_tok = train_ds.map(tokenize_batch, batched=True)
    dev_tok   = dev_ds.map(tokenize_batch, batched=True)

    # Rename Label -> labels for Trainer
    train_tok = train_tok.rename_column("Label", "labels")
    dev_tok   = dev_tok.rename_column("Label", "labels")

    # Drop unused columns if present
    cols_to_remove = [
        "Record-ID",
        "Class",
        "Transcript_PFT",
        "Transcript_CTD",
        "Transcript_SFT",
        "__index_level_0__",
    ]
    cols_to_remove = [c for c in cols_to_remove if c in train_tok.column_names]
    train_tok = train_tok.remove_columns(cols_to_remove)
    dev_tok   = dev_tok.remove_columns(cols_to_remove)

    train_tok.set_format("torch")
    dev_tok.set_format("torch")

    num_labels = df["Label"].nunique()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # Unique output directory per trial so checkpoint dirs don't collide
    model_dir = (
        f"checkpoints_random_search/"
        f"{transcript_col}_fold_{fold_idx}_trial_{trial_id}"
    )
    os.makedirs(model_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to=[],  # no wandb, tensorboard, etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Single-fold metrics (from Trainer.evaluate()):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Predictions on this fold's dev set
    preds_output = trainer.predict(dev_tok)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = dev_tok["labels"].numpy()

    metrics_simple = {
        "accuracy": metrics["eval_accuracy"],
        "macro_f1": metrics["eval_macro_f1"],
        "weighted_f1": metrics["eval_weighted_f1"],
        "precision_macro": metrics["eval_precision_macro"],
        "recall_macro": metrics["eval_recall_macro"],
        "precision_weighted": metrics["eval_precision_weighted"],
        "recall_weighted": metrics["eval_recall_weighted"],
        "model_dir": model_dir,
    }

    return metrics_simple, y_true, y_pred, model_dir


# -------------------------
# Random hyperparameter sampler
# -------------------------

def sample_hyperparams(rng: np.random.Generator) -> Dict:
    """
    Sample one random hyperparameter configuration.

    - learning_rate: discrete {1e-5, 2e-5, 3e-5, 5e-5}
    - batch_size: {8, 16}
    - num_train_epochs: {3, 4, 5}
    - max_len: {256, 384, 512}
    - weight_decay: uniform in [0.0, 0.1]
    - warmup_ratio: {0.0, 0.1, 0.2}
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
# Random search on a single transcript type
# -------------------------

def random_search_single_fold(
    df: pd.DataFrame,
    transcript_col: str,
    model_name: str = "bert-base-uncased",
    n_trials: int = 15,
    n_splits: int = 5,
    fold_to_use: int = 1,
    seed: int = 42,
    results_csv_path: str = "random_search_results.csv",
) -> pd.DataFrame:
    """
    Run random search over hyperparameters on a single fold for one transcript type.

    Saves:
      - all trials (hyperparams + metrics + model_dir) to results_csv_path

    Returns:
        results_df: DataFrame with one row per trial.
    """
    rng = np.random.default_rng(seed)

    all_results: List[Dict] = []
    best_macro_f1 = -1.0
    best_cfg: Dict = {}

    for trial in range(1, n_trials + 1):
        print(f"\n==============================")
        print(f" Random search trial {trial}/{n_trials} for {transcript_col}")
        print(f"==============================")

        cfg = sample_hyperparams(rng)
        print("Sampled hyperparams:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")

        metrics, _, _, model_dir = run_single_fold_bert(
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
            fold_to_use=fold_to_use,
            trial_id=trial,
        )

        trial_result = {
            **cfg,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "model_dir": model_dir,
        }
        all_results.append(trial_result)

        print(f"Trial {trial} metrics:")
        print(f"  accuracy           = {metrics['accuracy']:.4f}")
        print(f"  macro_f1           = {metrics['macro_f1']:.4f}")
        print(f"  weighted_f1        = {metrics['weighted_f1']:.4f}")
        print(f"  precision_macro    = {metrics['precision_macro']:.4f}")
        print(f"  recall_macro       = {metrics['recall_macro']:.4f}")
        print(f"  precision_weighted = {metrics['precision_weighted']:.4f}")
        print(f"  recall_weighted    = {metrics['recall_weighted']:.4f}")
        print(f"  model_dir          = {model_dir}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_cfg = trial_result
            print(">>> New best configuration found!")

        # Save intermediate results after each trial
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_csv_path, index=False)

    print("\n===== Random search complete for", transcript_col, "=====")
    print("Best configuration (by macro_f1):")
    for k, v in best_cfg.items():
        print(f"  {k}: {v}")

    results_df = pd.DataFrame(all_results).sort_values("macro_f1", ascending=False)
    return results_df


# -------------------------
# Main: run on ALL transcript types
# -------------------------

if __name__ == "__main__":
    # Path to your cleaned transcripts CSV
    TRANSCRIPTS_CSV = "data/transcripts_cleaned.csv"

    # Transcript columns to tune
    TRANSCRIPT_COLS = ["Transcript_PFT", "Transcript_CTD", "Transcript_SFT"]

    # Number of random trials per transcript type
    N_TRIALS = 15

    # Which fold to use for tuning (1-based index, 1..n_splits)
    FOLD_TO_USE = 1

    # Load one DataFrame per transcript type, dropping NaNs appropriately
    df_by_transcript = load_transcript_splits(
        TRANSCRIPTS_CSV,
        transcript_cols=TRANSCRIPT_COLS,
    )

    best_overall = []

    for transcript_col, df in df_by_transcript.items():
        print("\n############################################")
        print(f" Random search for transcript type: {transcript_col}")
        print("############################################")

        results_csv_path = f"random_search_{transcript_col}_fold{FOLD_TO_USE}.csv"

        results_df = random_search_single_fold(
            df=df,
            transcript_col=transcript_col,
            model_name="bert-base-uncased",
            n_trials=N_TRIALS,
            n_splits=5,
            fold_to_use=FOLD_TO_USE,
            seed=42,
            results_csv_path=results_csv_path,
        )

        # Grab the best row for this transcript type
        best_row = results_df.iloc[0].to_dict()
        best_row["transcript_col"] = transcript_col
        best_overall.append(best_row)

        print(f"\nTop 5 configs for {transcript_col}:")
        print(results_df.head())

    # Compare best configs across transcript types
    best_df = pd.DataFrame(best_overall).sort_values("macro_f1", ascending=False)
    best_df.to_csv("random_search_best_per_transcript.csv", index=False)

    print("\n===== Best config per transcript type (sorted by macro_f1) =====")
    print(best_df)
