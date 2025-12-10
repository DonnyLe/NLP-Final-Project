"""

This file is for evaluating pretrained language models (transformers) for dementia classification

Quick start guide: 
    - To run the fine-tuning and validation code, we utilized the Northeastern GPUs
    - Tried to run the code natively on M1 Macbook Air, but was running into an error (it may work on another computer though)
    - Command: python -m pretrained_lm.plm_random_search (on the project root)
    - If you want to change the model do: PLM_MODEL_NAME=model_name python -m pretrained_lm.plm_random_search
    - model_name can be roberta-large or bert-base-uncased

On the Northeastern GPU cluster: 
    - Instructions to SSH: https://rc-docs.northeastern.edu/en/latest/gpus/index.html
    - Clone repo
    - sbatch run_roberta_random.sh
    or 
    - sbatch run_bert_random.sh

"""


import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from transcript_preprocessing import load_transcript_splits

from pretrained_lm.plm_utils import random_search_kfold

CLASS_NAMES = ["HC", "MCI", "Dementia"]

# Model: 

# MODEL_NAME = "bert-base-uncased"
# MODEL_NAME ='roberta-large'
MODEL_NAME = os.environ.get("PLM_MODEL_NAME", "bert-base-uncased")
MODEL_TAG = MODEL_NAME.split("/")[-1]



if __name__ == "__main__":
    TRANSCRIPTS_CSV = "data/transcripts_cleaned.csv"

    # upsampling and class weight flags  
    USE_UPSAMPLING = False
    USE_CLASS_WEIGHTS = True

    if USE_UPSAMPLING and USE_CLASS_WEIGHTS:
        raise ValueError("Set only one of USE_UPSAMPLING / USE_CLASS_WEIGHTS to True.")

    # num of trials and folds
    N_TRIALS = 10
    N_SPLITS = 5

        # Load raw data once
    raw_df = pd.read_csv(TRANSCRIPTS_CSV)

   

    df_by_transcript = load_transcript_splits(
        TRANSCRIPTS_CSV,
    )

    best_overall: List[Dict] = []


    # filename suffixes 
    if USE_UPSAMPLING:
        suffix = "_upsampled"
    elif USE_CLASS_WEIGHTS:
        suffix = "_weighted"
    else:
        suffix = "_baseline"

    for transcript_col, df in df_by_transcript.items():
        print(
            f"(upsampling={USE_UPSAMPLING}, class_weights={USE_CLASS_WEIGHTS}, model={MODEL_NAME})"
        )

        results_csv_path = f"{MODEL_TAG}_random_search_{transcript_col}_folds{suffix}.csv"

        results_df, best_cfg, y_true_best, y_pred_best = random_search_kfold(
            df=df,
            transcript_col=transcript_col,
            label_col="Label",
            model_name=MODEL_NAME,
            model_tag=MODEL_TAG,
            n_trials=N_TRIALS,
            n_splits=N_SPLITS,
            seed=42,
            results_csv_path=results_csv_path,
            use_upsampling=USE_UPSAMPLING,
            use_class_weights=USE_CLASS_WEIGHTS,
        )

        best_row = best_cfg.copy()
        best_row["transcript_col"] = transcript_col
        best_overall.append(best_row)
        print()
        print(f"Top configs for {transcript_col}:")
        print(results_df.head())

        # confusion matrix for best trial of this transcript type
        cm = confusion_matrix(y_true_best, y_pred_best, labels=[0, 1, 2])

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{MODEL_TAG} Confusion Matrix: {transcript_col}{suffix}")

        cm_filename = f"{MODEL_TAG}_confusion_matrix_{transcript_col}{suffix}.png"
        fig.savefig(cm_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        report = classification_report(
            y_true_best,
            y_pred_best,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        print()
        print(f"Classification report for {transcript_col}{suffix}:")
        print(report)

        report_filename = f"{MODEL_TAG}_classification_report_{transcript_col}{suffix}.txt"
        with open(report_filename, "w") as f:
            f.write(f"Classification report for {transcript_col}{suffix} (model={MODEL_NAME})\n\n")
            f.write(report)
            f.write("\n")
            f.write("\nBest trial CV metrics:\n")
            f.write(f"accuracy = {best_cfg['accuracy']}\n")
            f.write(f"macro_f1 = {best_cfg['macro_f1']}\n")
            f.write(f"weighted_f1 = {best_cfg['weighted_f1']}\n")

    best_df = pd.DataFrame(best_overall).sort_values("macro_f1", ascending=False)
    best_df.to_csv(f"{MODEL_TAG}_random_search_best_per_transcript{suffix}.csv", index=False)

    summary_path = f"{MODEL_TAG}_random_search_summary{suffix}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Best config per transcript type (sorted by macro_f1) for model={MODEL_NAME}:\n\n")
        f.write(best_df.to_string(index=False))
        f.write("\n")

    print("Best configs table:")
    print(best_df)
