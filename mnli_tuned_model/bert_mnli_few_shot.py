'''
Note: This code was scratched in favor of RoBERTa experimentation 
since choosing an MNLI model did not seem to show any performance increases 

To run: 
    python -m mnli_tuned_model.bert_mnli_few_shot

Like the other transformer-based modals, we ran this on the Northeastern GPUs
Did not test running this locally 
'''



import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)

from transcript_preprocessing import load_transcript_splits

# 🔁 Reuse shared PLM helpers
from pretrained_lm.plm_utils import (
    compute_metrics_from_labels,
    compute_metrics_for_logits,
    tokenize_batch_factory,
    compute_class_weights_from_df,
    WeightedCETrainer,
)

CLASS_NAMES = ["HC", "MCI", "Dementia"]

BASE_MODEL_NAME = "roberta-large-mnli"

# Few-shot size (per class)
N_SHOT_PER_CLASS = 20  # e.g., at most 20 examples per label

# Training hyperparameters
MAX_LEN = 256
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

OUTPUT_DIR = "few_shot_mnli_outputs"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------
# Helpers
# -------------------------

def make_few_shot_subset(
    df: pd.DataFrame,
    label_col: str = "Label",
    per_class: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a small few-shot subset with up to `per_class` examples per label.
    If a class has fewer than per_class examples, we keep all of them.
    """
    rng = np.random.default_rng(seed)

    few_shot_frames = []
    labels = sorted(df[label_col].unique())

    for lbl in labels:
        class_df = df[df[label_col] == lbl]
        if len(class_df) <= per_class:
            sampled = class_df
        else:
            sampled = class_df.sample(
                n=per_class,
                random_state=int(rng.integers(0, 1_000_000)),
            )
        few_shot_frames.append(sampled)

    few_shot_df = pd.concat(few_shot_frames, axis=0)
    few_shot_df = few_shot_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return few_shot_df

# Few-shot training for one transcript type
def run_few_shot_for_transcript(
    df: pd.DataFrame,
    transcript_col: str,
    model_name: str = BASE_MODEL_NAME,
    n_shot_per_class: int = N_SHOT_PER_CLASS,
    seed: int = 42,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Few-shot fine-tune MNLI-initialized model on a small subset
    for a single transcript column. Returns metrics, y_true, y_pred on dev set.
    """
    print(f"\n=== FEW-SHOT TRAINING for {transcript_col} ===")
    print(f"Base model: {model_name}")
    print(f"Few-shot: up to {n_shot_per_class} examples per class")

    # Keep rows with non-null transcript + label
    df_clean = df.dropna(subset=[transcript_col, "Label"]).copy()
    df_clean = df_clean.reset_index(drop=True)

    # Build few-shot subset
    few_shot_df = make_few_shot_subset(
        df_clean,
        label_col="Label",
        per_class=n_shot_per_class,
        seed=seed,
    )
    print(f"Few-shot subset size: {len(few_shot_df)} rows")

    # Train/dev split (stratified)
    train_df, dev_df = train_test_split(
        few_shot_df,
        test_size=0.2,
        random_state=seed,
        stratify=few_shot_df["Label"],
    )
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)

    print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}")

    class_weights = compute_class_weights_from_df(train_df, label_col="Label")
    print("Class weights:", class_weights.tolist())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASS_NAMES),
    )

    tokenize_batch = tokenize_batch_factory(transcript_col, tokenizer, MAX_LEN)

    train_ds = Dataset.from_pandas(train_df)
    dev_ds = Dataset.from_pandas(dev_df)

    train_tok = train_ds.map(tokenize_batch, batched=True)
    dev_tok = dev_ds.map(tokenize_batch, batched=True)

    train_tok = train_tok.rename_column("Label", "labels")
    dev_tok = dev_tok.rename_column("Label", "labels")

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
    dev_tok = dev_tok.remove_columns(cols_to_remove)

    train_tok.set_format("torch")
    dev_tok.set_format("torch")

    output_dir = os.path.join(OUTPUT_DIR, f"checkpoints_{transcript_col}")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=False,
        logging_steps=10,
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
    eval_results = trainer.evaluate()
    print(f"Dev evaluation metrics for {transcript_col}:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    preds_output = trainer.predict(dev_tok)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = np.array(dev_tok["labels"])

    metrics = compute_metrics_from_labels(y_true, y_pred)
    return metrics, y_true, y_pred

# Main
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TRANSCRIPTS_CSV = "data/transcripts_cleaned.csv"
    raw_df = pd.read_csv(TRANSCRIPTS_CSV)

    # Build Transcript_ALL like in your other scripts
    raw_df["Transcript_ALL"] = (
        "[PFT] " + raw_df["Transcript_PFT"].fillna("") + " "
        + "[CTD] " + raw_df["Transcript_CTD"].fillna("") + " "
        + "[SFT] " + raw_df["Transcript_SFT"].fillna("")
    )

    tmp_csv = "data/transcripts_with_all.csv"
    raw_df.to_csv(tmp_csv, index=False)

    TRANSCRIPT_COLS = [
        "Transcript_PFT",
        "Transcript_CTD",
        "Transcript_SFT",
        "Transcript_ALL",
    ]

    df_by_transcript = load_transcript_splits(
        tmp_csv,
        transcript_cols=TRANSCRIPT_COLS,
    )

    summary_rows: List[Dict] = []

    for transcript_col, df in df_by_transcript.items():
        metrics, y_true, y_pred = run_few_shot_for_transcript(
            df=df,
            transcript_col=transcript_col,
            model_name=BASE_MODEL_NAME,
            n_shot_per_class=N_SHOT_PER_CLASS,
            seed=42,
        )

        print(f"\nFinal few-shot metrics for {transcript_col}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        summary_row = {"transcript_col": transcript_col, **metrics}
        summary_rows.append(summary_row)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Few-shot MNLI-initialized BERT - {transcript_col}")

        cm_filename = os.path.join(
            OUTPUT_DIR, f"fewshot_confusion_matrix_{transcript_col}.png"
        )
        fig.savefig(cm_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        print(f"\nClassification report for {transcript_col}:")
        print(report)

        report_filename = os.path.join(
            OUTPUT_DIR, f"fewshot_classification_report_{transcript_col}.txt"
        )
        with open(report_filename, "w") as f:
            f.write(f"Few-shot MNLI-initialized BERT classification report for {transcript_col}\n\n")
            f.write(report)
            f.write("\n")
            f.write("\nMetrics:\n")
            for k, v in metrics.items():
                f.write(f"{k} = {v}\n")

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_csv = os.path.join(OUTPUT_DIR, "fewshot_summary_by_transcript.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("Summary:")
    print(summary_df)
