'''
Note: This code is scratched due to the poor performance. 
Accidentally removed the data, but generally, the modal was essentially guessing randomly 

To run: 
    python -m mnli_tuned_model.bert_mnli_zero

Like the other transformer-based modals, we ran this on the Northeastern GPUs.
This was done using the "run_bert_mnli_zero_shot.sh" bash script after SSHing into the cluster.
Did not test running this locally 


Used this code as a reference: https://joeddav.github.io/blog/2020/05/29/ZSL.html
'''



import os
from typing import Dict, List, Tuple

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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from transcript_preprocessing import load_transcript_splits

CLASS_NAMES = ["HC", "MCI", "Dementia"]

# Avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Pretrained NLI model: BERT-base fine-tuned on MNLI
NLI_MODEL_NAME = "roberta-large-mnli"

# Hypothesis templates for each class (labels 0,1,2)
CLASS_HYPOTHESES = {
    0: "This transcript is from a healthy control participant with no cognitive impairment.",
    1: "This transcript is from a patient with mild cognitive impairment.",
    2: "This transcript is from a patient with dementia.",
}


# -------------------------
# Metric helper
# -------------------------

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



# Zero-shot prediction with BERT-MNLI

def zero_shot_predict(
    texts: List[str],
    tokenizer,
    model,
    max_len: int = 256,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Zero-shot classification using an NLI model.
    For each text, we create a pair (premise=text, hypothesis=class_hypothesis)
    for each class and pick the class with the highest entailment logit.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds: List[int] = []

    num_texts = len(texts)
    num_classes = len(CLASS_HYPOTHESES)

    with torch.no_grad():
        for start in range(0, num_texts, batch_size):
            end = min(start + batch_size, num_texts)
            batch_texts = texts[start:end]

            # for each text in batch, build num_classes pairs
            #  then pick the best class per example
            entailment_scores = []

            for class_idx in range(num_classes):
                hyp = CLASS_HYPOTHESES[class_idx]

                # tokenize premise–hypothesis pairs
                encoded = tokenizer(
                    batch_texts,
                    [hyp] * len(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(device)

                outputs = model(**encoded)
                # For BERT-MNLI, label mapping is typically:
                # 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
                logits = outputs.logits  # [batch, 3]
                # We only care about the ENTAILMENT logit for each pair
                entail_logit = logits[:, 2]  # [batch]
                entailment_scores.append(entail_logit.cpu().numpy())

            # entailment_scores shape: [num_classes, batch_size]
            # -> transpose to [batch_size, num_classes]
            batch_scores = np.stack(entailment_scores, axis=0).T
            # Argmax over classes
            batch_preds = np.argmax(batch_scores, axis=1)
            all_preds.extend(batch_preds.tolist())

    return np.array(all_preds, dtype=int)


# zero-shot evaluation for one transcript type
def evaluate_zero_shot_for_transcript(
    df: pd.DataFrame,
    transcript_col: str,
    tokenizer,
    model,
    max_len: int = 256,
    batch_size: int = 8,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Run zero-shot BERT-MNLI on a single transcript column and compute metrics.
    Assumes df has columns:
      - transcript_col: text
      - "Label": integer labels 0..(num_classes-1)
    """

    # Drop rows with missing transcript or label
    df_clean = df.dropna(subset=[transcript_col, "Label"]).copy()
    df_clean = df_clean.reset_index(drop=True)

    texts = df_clean[transcript_col].tolist()
    y_true = df_clean["Label"].astype(int).to_numpy()

    y_pred = zero_shot_predict(
        texts,
        tokenizer=tokenizer,
        model=model,
        max_len=max_len,
        batch_size=batch_size,
    )

    metrics = compute_metrics_from_labels(y_true, y_pred)
    return metrics, y_true, y_pred


# Main: run zero-shot on ALL transcript types
if __name__ == "__main__":
    TRANSCRIPTS_CSV = "data/transcripts_cleaned.csv"
    raw_df = pd.read_csv(TRANSCRIPTS_CSV)
    TRANSCRIPT_COLS = [
        "Transcript_PFT",
        "Transcript_CTD",
        "Transcript_SFT",
    ]

    df_by_transcript = load_transcript_splits(
        TRANSCRIPTS_CSV,
        transcript_cols=TRANSCRIPT_COLS,
    )


    print("Loading NLI model:", NLI_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)

    summary_rows = []
    os.makedirs("zero_shot_outputs", exist_ok=True)

    for transcript_col, df in df_by_transcript.items():
        metrics, y_true, y_pred = evaluate_zero_shot_for_transcript(
            df=df,
            transcript_col=transcript_col,
            tokenizer=tokenizer,
            model=model,
            max_len=256,
            batch_size=8,
        )

        row = {
            "transcript_col": transcript_col,
            **metrics,
        }
        summary_rows.append(row)

        print(f"\nMetrics for {transcript_col}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Zero-shot BERT-MNLI - {transcript_col}")

        cm_filename = os.path.join("zero_shot_outputs", f"zs_confusion_matrix_{transcript_col}.png")
        fig.savefig(cm_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )

        print(f"\nClassification report for {transcript_col}:")
        print(report)

        report_filename = os.path.join("zero_shot_outputs", f"zs_classification_report_{transcript_col}.txt")
        with open(report_filename, "w") as f:
            f.write(f"Zero-shot BERT-MNLI classification report for {transcript_col}\n\n")
            f.write(report)
            f.write("\n")
            f.write("\nMetrics:\n")
            for k, v in metrics.items():
                f.write(f"{k} = {v}\n")

    # summary across transcript types
    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_df.to_csv("zero_shot_outputs/zs_summary_by_transcript.csv", index=False)

    print("\nFinished zero-shot evaluation for all transcript types.")
    print("Summary:")
    print(summary_df)
