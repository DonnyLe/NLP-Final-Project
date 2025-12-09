# plm_cascade.py

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from transcript_preprocessing import load_transcript_splits, get_stratified_kfold_splits

from plm_utils import (
    compute_metrics_from_labels,
    random_search_kfold,
    train_binary_single_split,
)

# 3-class names (original task)
CLASS_NAMES = ["HC", "MCI", "Dementia"]

# ------------------------------------------------------------------
# MODEL CONFIG
# ------------------------------------------------------------------
# You can override via env var: PLM_MODEL_NAME=roberta-large
MODEL_NAME = "bert-base-uncased"
MODEL_TAG = MODEL_NAME.split("/")[-1]


def extract_hparams_from_best_cfg(best_cfg: Dict) -> Dict:
    keys = ["learning_rate", "batch_size", "num_train_epochs",
            "max_len", "weight_decay", "warmup_ratio"]
    return {k: best_cfg[k] for k in keys}


def run_cascade_kfold(
    base_df: pd.DataFrame,
    transcript_col: str,
    best_cfg_hc_nonhc: Dict,
    best_cfg_mci_dem: Dict,
    n_splits: int,
    seed: int,
    model_name: str,
    use_class_weights: bool,
    use_upsampling: bool,
) -> None:
    """
    Full K-fold cascaded evaluation:
      - For each fold:
          * train HC vs Non-HC on train split
          * train MCI vs Dementia on Non-HC subset of train split
          * run cascade on dev split to get 3-class predictions
      - Aggregate metrics and confusion matrix over all folds.
    """
    print(f"\nRunning FULL K-fold cascaded evaluation for {transcript_col} with model={model_name}")

    y_all = base_df["Label_3cls"].values

    folds = list(
        get_stratified_kfold_splits(
            base_df,
            transcript_col=transcript_col,
            label_col="Label_3cls",
            n_splits=n_splits,
            seed=seed,
        )
    )

    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []

    hparams_A = extract_hparams_from_best_cfg(best_cfg_hc_nonhc)
    hparams_B = extract_hparams_from_best_cfg(best_cfg_mci_dem)

    for fold_idx, train_idx, dev_idx in folds:
        print(f"\n--- Cascade fold {fold_idx} ---")

        train_df = base_df.iloc[train_idx].reset_index(drop=True)
        dev_df = base_df.iloc[dev_idx].reset_index(drop=True)

        # Model A: HC vs Non-HC
        train_A = train_df.copy()
        dev_A = dev_df.copy()
        train_A["Label_bin"] = (train_A["Class"] != "HC").astype(int)
        dev_A["Label_bin"] = (dev_A["Class"] != "HC").astype(int)

        preds_A_dev = train_binary_single_split(
            train_df=train_A,
            dev_df=dev_A,
            transcript_col=transcript_col,
            label_col="Label_bin",
            model_name=model_name,
            max_len=hparams_A["max_len"],
            learning_rate=hparams_A["learning_rate"],
            num_train_epochs=hparams_A["num_train_epochs"],
            batch_size=hparams_A["batch_size"],
            weight_decay=hparams_A["weight_decay"],
            warmup_ratio=hparams_A["warmup_ratio"],
            use_class_weights=use_class_weights,
            use_upsampling=use_upsampling,
            seed=seed + fold_idx,
            tmp_dir_tag=f"{MODEL_TAG}_cascade_A_fold{fold_idx}",
        )

        # Model B: MCI vs Dementia on Non-HC subset of train
        train_B = train_df[train_df["Class"] != "HC"].copy()
        dev_B_full = dev_df.copy()

        train_B["Label_bin"] = (train_B["Class"] == "Dementia").astype(int)
        dev_B_full["Label_bin"] = np.where(
            dev_B_full["Class"] == "Dementia",
            1,
            0,
        )

        preds_B_dev_full = train_binary_single_split(
            train_df=train_B,
            dev_df=dev_B_full,
            transcript_col=transcript_col,
            label_col="Label_bin",
            model_name=model_name,
            max_len=hparams_B["max_len"],
            learning_rate=hparams_B["learning_rate"],
            num_train_epochs=hparams_B["num_train_epochs"],
            batch_size=hparams_B["batch_size"],
            weight_decay=hparams_B["weight_decay"],
            warmup_ratio=hparams_B["warmup_ratio"],
            use_class_weights=use_class_weights,
            use_upsampling=use_upsampling,
            seed=seed + 100 + fold_idx,
            tmp_dir_tag=f"{MODEL_TAG}_cascade_B_fold{fold_idx}",
        )

        # Build cascaded 3-class predictions for this fold's dev set
        y_true_3cls_fold = dev_df["Label_3cls"].values
        cascade_preds_fold: List[int] = []

        for i in range(len(dev_df)):
            pred_A = preds_A_dev[i]
            if pred_A == 0:
                cascade_preds_fold.append(0)  # HC
            else:
                # Non-HC -> use Model B
                pred_B = preds_B_dev_full[i]  # 0 = MCI, 1 = Dementia
                if pred_B == 0:
                    cascade_preds_fold.append(1)  # MCI
                else:
                    cascade_preds_fold.append(2)  # Dementia

        all_y_true.append(y_true_3cls_fold)
        all_y_pred.append(np.array(cascade_preds_fold))

    y_true_all = np.concatenate(all_y_true)
    cascade_preds_3cls = np.concatenate(all_y_pred)

    metrics_3cls = compute_metrics_from_labels(y_true_all, cascade_preds_3cls)
    print("\nK-fold Cascaded 3-class metrics:")
    print("  accuracy:", metrics_3cls["accuracy"])
    print("  macro_f1:", metrics_3cls["macro_f1"])
    print("  weighted_f1:", metrics_3cls["weighted_f1"])

    cm = confusion_matrix(y_true_all, cascade_preds_3cls, labels=[0, 1, 2])

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Cascaded {MODEL_TAG} Confusion Matrix (K-fold) - {transcript_col}")
    fig.savefig(f"{MODEL_TAG}_cascade_kfold_confusion_matrix_{transcript_col}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        y_true_all,
        cascade_preds_3cls,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print("\nCascaded 3-class classification report (K-fold):")
    print(report)

    with open(f"{MODEL_TAG}_cascade_kfold_classification_report_{transcript_col}.txt", "w") as f:
        f.write(f"Cascaded 3-class K-fold report for {transcript_col} (model={model_name})\n\n")
        f.write(report)
        f.write("\n")
        f.write("Aggregated K-fold metrics:\n")
        f.write(f"accuracy = {metrics_3cls['accuracy']}\n")
        f.write(f"macro_f1 = {metrics_3cls['macro_f1']}\n")
        f.write(f"weighted_f1 = {metrics_3cls['weighted_f1']}\n")


# =========================
# Main
# =========================

if __name__ == "__main__":
    TRANSCRIPTS_CSV = "data/transcripts_cleaned.csv"

    # We will run on these transcript types ONLY (not Transcript_ALL)
    TRANSCRIPT_COLS_TO_RUN = [
        "Transcript_PFT",
        "Transcript_CTD",
        "Transcript_SFT",
    ]

    # Random search + K-fold settings for the binary tasks
    N_TRIALS = 10
    N_SPLITS = 5
    SEED = 42

    USE_UPSAMPLING = False
    USE_CLASS_WEIGHTS = True

    raw_df = pd.read_csv(TRANSCRIPTS_CSV)

    # Build combined transcript (not used here, but needed for load_transcript_splits)
    raw_df["Transcript_ALL"] = (
        "[PFT] " + raw_df["Transcript_PFT"].fillna("") + " "
        + "[CTD] " + raw_df["Transcript_CTD"].fillna("") + " "
        + "[SFT] " + raw_df["Transcript_SFT"].fillna("")
    )

    tmp_csv = "data/transcripts_with_all_for_cascade.csv"
    raw_df.to_csv(tmp_csv, index=False)

    df_by_transcript = load_transcript_splits(
        tmp_csv,
        transcript_cols=[
            "Transcript_PFT",
            "Transcript_CTD",
            "Transcript_SFT",
            "Transcript_ALL",
        ],
    )

    for transcript_col in TRANSCRIPT_COLS_TO_RUN:
        if transcript_col not in df_by_transcript:
            raise ValueError(f"{transcript_col} not found in df_by_transcript keys.")

        print(f"\n==============================")
        print(f"Running cascaded experiments for {transcript_col} (model={MODEL_NAME})")
        print(f"==============================")

        base_df = df_by_transcript[transcript_col].copy()

        # Preserve original 3-class numeric labels
        base_df["Label_3cls"] = base_df["Label"]

        # -------------------
        # Model A: HC vs Non-HC (binary)
        # -------------------
        df_hc_nonhc = base_df.copy()
        df_hc_nonhc["Label_bin"] = (df_hc_nonhc["Class"] != "HC").astype(int)

        print(f"\n=== Binary Task 1: HC vs Non-HC ({transcript_col}) ===")
        results_A, best_cfg_A, y_true_A, y_pred_A = random_search_kfold(
            df=df_hc_nonhc,
            transcript_col=transcript_col,
            label_col="Label_bin",
            model_name=MODEL_NAME,
            model_tag=MODEL_TAG,
            n_trials=N_TRIALS,
            n_splits=N_SPLITS,
            seed=SEED,
            results_csv_path=f"{MODEL_TAG}_random_search_HC_vs_NonHC_{transcript_col}.csv",
            use_upsampling=USE_UPSAMPLING,
            use_class_weights=USE_CLASS_WEIGHTS,
        )

        print("\nTop configs for HC vs Non-HC:")
        print(results_A.head())

        cm_A = confusion_matrix(y_true_A, y_pred_A, labels=[0, 1])
        fig, ax = plt.subplots()
        disp_A = ConfusionMatrixDisplay(
            confusion_matrix=cm_A,
            display_labels=["HC", "Non-HC"],
        )
        disp_A.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix - HC vs Non-HC - {transcript_col} ({MODEL_TAG})")
        fig.savefig(f"{MODEL_TAG}_confusion_matrix_HC_vs_NonHC_{transcript_col}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        report_A = classification_report(
            y_true_A,
            y_pred_A,
            target_names=["HC", "Non-HC"],
            zero_division=0,
        )
        with open(f"{MODEL_TAG}_classification_report_HC_vs_NonHC_{transcript_col}.txt", "w") as f:
            f.write(f"Binary HC vs Non-HC report for {transcript_col} (model={MODEL_NAME})\n\n")
            f.write(report_A)
            f.write("\n")

        # -------------------
        # Model B: MCI vs Dementia (binary, Non-HC subset)
        # -------------------
        df_mci_dem = base_df[base_df["Class"] != "HC"].copy()
        df_mci_dem["Label_bin"] = (df_mci_dem["Class"] == "Dementia").astype(int)

        print(f"\n=== Binary Task 2: MCI vs Dementia ({transcript_col}) ===")
        results_B, best_cfg_B, y_true_B, y_pred_B = random_search_kfold(
            df=df_mci_dem,
            transcript_col=transcript_col,
            label_col="Label_bin",
            model_name=MODEL_NAME,
            model_tag=MODEL_TAG,
            n_trials=N_TRIALS,
            n_splits=N_SPLITS,
            seed=SEED,
            results_csv_path=f"{MODEL_TAG}_random_search_MCI_vs_Dem_{transcript_col}.csv",
            use_upsampling=USE_UPSAMPLING,
            use_class_weights=USE_CLASS_WEIGHTS,
        )

        print("\nTop configs for MCI vs Dementia:")
        print(results_B.head())

        cm_B = confusion_matrix(y_true_B, y_pred_B, labels=[0, 1])
        fig, ax = plt.subplots()
        disp_B = ConfusionMatrixDisplay(
            confusion_matrix=cm_B,
            display_labels=["MCI", "Dementia"],
        )
        disp_B.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix - MCI vs Dementia - {transcript_col} ({MODEL_TAG})")
        fig.savefig(f"{MODEL_TAG}_confusion_matrix_MCI_vs_Dem_{transcript_col}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        report_B = classification_report(
            y_true_B,
            y_pred_B,
            target_names=["MCI", "Dementia"],
            zero_division=0,
        )
        with open(f"{MODEL_TAG}_classification_report_MCI_vs_Dem_{transcript_col}.txt", "w") as f:
            f.write(f"Binary MCI vs Dementia report for {transcript_col} (model={MODEL_NAME})\n\n")
            f.write(report_B)
            f.write("\n")

        # -------------------
        # FULL K-fold cascaded 3-class experiment using best configs
        # -------------------
        run_cascade_kfold(
            base_df=base_df,
            transcript_col=transcript_col,
            best_cfg_hc_nonhc=best_cfg_A,
            best_cfg_mci_dem=best_cfg_B,
            n_splits=N_SPLITS,
            seed=SEED,
            model_name=MODEL_NAME,
            use_class_weights=USE_CLASS_WEIGHTS,
            use_upsampling=USE_UPSAMPLING,
        )

    print(f"\nAll cascaded K-fold experiments finished for PFT / CTD / SFT with {MODEL_NAME}.")
