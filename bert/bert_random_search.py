import os
import shutil
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
    fold_to_use: int = 1,
    trial_id: int = 0,
    save_model: bool = False,  # NEW: only save if explicitly requested
    final_model_dir: str = None,  # NEW: where to save the best model
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
        label_col="Label",
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

    # Temporary directory for training
    temp_model_dir = f"temp_checkpoints/trial_{trial_id}"
    os.makedirs(temp_model_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=temp_model_dir,
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
        save_total_limit=1,  # Only keep 1 checkpoint during training
        dataloader_num_workers=0,
        report_to=[],
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
        "model_dir": final_model_dir if save_model else temp_model_dir,
    }

    # NEW: Only save model if this is the best one
    if save_model and final_model_dir:
        print(f">>> Saving best model to {final_model_dir}")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
    
    # Clean up temporary directory
    if os.path.exists(temp_model_dir):
        shutil.rmtree(temp_model_dir)
        print(f"Cleaned up temporary directory: {temp_model_dir}")

    return metrics_simple, y_true, y_pred, final_model_dir if save_model else "not_saved"


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
    Only saves the best model based on macro_f1.

    Saves:
      - all trials (hyperparams + metrics + model_dir) to results_csv_path
      - ONLY the best model to disk

    Returns:
        results_df: DataFrame with one row per trial.
    """
    rng = np.random.default_rng(seed)

    all_results: List[Dict] = []
    best_macro_f1 = -1.0
    best_cfg: Dict = {}
    best_trial_id = -1

    # First pass: run all trials without saving models
    print("\n" + "="*60)
    print(f" Phase 1: Running {n_trials} trials for {transcript_col}")
    print(f" Models will NOT be saved yet - finding best config first")
    print("="*60)

    for trial in range(1, n_trials + 1):
        print(f"\n{'='*30}")
        print(f" Trial {trial}/{n_trials} for {transcript_col}")
        print(f"{'='*30}")

        cfg = sample_hyperparams(rng)
        print("Sampled hyperparams:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")

        metrics, _, _, _ = run_single_fold_bert(
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
            save_model=False,  # Don't save yet
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

        print(f"Trial {trial} metrics:")
        print(f"  accuracy           = {metrics['accuracy']:.4f}")
        print(f"  macro_f1           = {metrics['macro_f1']:.4f}")
        print(f"  weighted_f1        = {metrics['weighted_f1']:.4f}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_cfg = {**trial_result}
            best_trial_id = trial
            print(">>> New best configuration found!")

        # Save intermediate results after each trial
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_csv_path, index=False)

    # Second pass: retrain ONLY the best model and save it
    print("\n" + "="*60)
    print(f" Phase 2: Retraining and saving ONLY the best model")
    print(f" Best trial: {best_trial_id} with macro_f1 = {best_macro_f1:.4f}")
    print("="*60)

    best_model_dir = f"best_models/{transcript_col}_fold_{fold_to_use}"
    os.makedirs(best_model_dir, exist_ok=True)

    print("\nRetraining best configuration:")
    for k, v in best_cfg.items():
        if k != 'trial_id':
            print(f"  {k}: {v}")

    final_metrics, _, _, saved_dir = run_single_fold_bert(
        df=df,
        transcript_col=transcript_col,
        model_name=model_name,
        max_len=best_cfg["max_len"],
        learning_rate=best_cfg["learning_rate"],
        num_train_epochs=best_cfg["num_train_epochs"],
        batch_size=best_cfg["batch_size"],
        weight_decay=best_cfg["weight_decay"],
        warmup_ratio=best_cfg["warmup_ratio"],
        n_splits=n_splits,
        seed=seed,
        fold_to_use=fold_to_use,
        trial_id=best_trial_id,
        save_model=True,
        final_model_dir=best_model_dir,
    )

    # Update the best config with the model directory
    best_cfg["model_dir"] = saved_dir

    print(f"\n✓ Best model saved to: {best_model_dir}")
    print(f"✓ All trial results saved to: {results_csv_path}")

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
    N_TRIALS = 10

    # Which fold to use for tuning (1-based index, 1..n_splits)
    FOLD_TO_USE = 1

    # Load one DataFrame per transcript type, dropping NaNs appropriately
    df_by_transcript = load_transcript_splits(
        TRANSCRIPTS_CSV,
        transcript_cols=TRANSCRIPT_COLS,
    )

    best_overall = []

    for transcript_col, df in df_by_transcript.items():
        print("\n" + "#"*60)
        print(f" Random search for transcript type: {transcript_col}")
        print("#"*60)

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

    print("\n" + "="*60)
    print(" FINAL SUMMARY: Best config per transcript type")
    print("="*60)
    print(best_df)
    print("\n✓ All best models saved in: best_models/")
    print("✓ Summary saved to: random_search_best_per_transcript.csv")