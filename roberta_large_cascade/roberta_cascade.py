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
from sklearn.model_selection import train_test_split
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

# 3-class names (original task)
CLASS_NAMES = ["HC", "MCI", "Dementia"]

# Backbone model
MODEL_NAME = "roberta-large"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================
# Helpers: tokenization + metrics
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


def compute_metrics_from_labels(labels, preds) -> Dict[str, float]:
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
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics_from_labels(labels, preds)


# =========================
# Upsampling + class weights
# =========================

def upsample_train_df(
    train_df: pd.DataFrame,
    label_col: str = "Label",
    seed: Optional[int] = None,
) -> pd.DataFrame:
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
# K-fold training for random search (binary or 3-class)
# =========================

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
    label_col: str = "Label",
    use_upsampling: bool = False,
    use_class_weights: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if use_upsampling and use_class_weights:
        raise ValueError("Choose either upsampling OR class-weighted loss, not both.")

    suffix = ""
    if use_upsampling:
        suffix = " (upsampling)"
    elif use_class_weights:
        suffix = " (class-weighted)"

    print(f"  Fold {fold_idx}, trial {trial_id}{suffix}")

    tokenize_batch = tokenize_batch_factory(transcript_col, tokenizer, max_len)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    dev_df = df.iloc[dev_idx].reset_index(drop=True)

    if use_upsampling:
        train_df = upsample_train_df(train_df, label_col=label_col, seed=seed + fold_idx)

    if use_class_weights:
        class_weights = compute_class_weights_from_df(train_df, label_col=label_col)
    else:
        class_weights = None

    # HuggingFace Dataset
    train_ds = Dataset.from_pandas(train_df)
    dev_ds = Dataset.from_pandas(dev_df)

    train_tok = train_ds.map(tokenize_batch, batched=True)
    dev_tok = dev_ds.map(tokenize_batch, batched=True)

    # Rename label_col -> labels
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

    num_labels = df[label_col].nunique()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    output_dir = "bert_tmp_cascade"
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
    label_col: str = "Label",
    use_upsampling: bool = False,
    use_class_weights: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
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
            label_col=label_col,
            use_upsampling=use_upsampling,
            use_class_weights=use_class_weights,
        )
        all_y_true.append(y_true_fold)
        all_y_pred.append(y_pred_fold)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    metrics = compute_metrics_from_labels(y_true_all, y_pred_all)
    return metrics, y_true_all, y_pred_all


# =========================
# Random search
# =========================

def sample_hyperparams(rng: np.random.Generator) -> Dict:
    """
    Sample hyperparameters tuned for a large model (roberta-large).
    """
    # Large models: keep LR small
    learning_rate = float(rng.choice([1e-5, 1.5e-5, 2e-5]))
    # Smaller batches for memory
    batch_size = int(rng.choice([2, 4]))
    # Moderate epochs to limit overfitting
    num_train_epochs = int(rng.choice([3, 4]))
    # Slightly shorter sequences for memory
    max_len = int(rng.choice([256, 384]))
    # Mild weight decay
    weight_decay = float(rng.uniform(0.0, 0.05))
    # Simple warmup options
    warmup_ratio = float(rng.choice([0.0, 0.1]))

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_train_epochs": num_train_epochs,
        "max_len": max_len,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
    }


def random_search_kfold(
    df: pd.DataFrame,
    transcript_col: str,
    label_col: str,
    model_name: str = MODEL_NAME,
    n_trials: int = 10,
    n_splits: int = 5,
    seed: int = 42,
    results_csv_path: str = "bert_random_search_binary.csv",
    use_upsampling: bool = False,
    use_class_weights: bool = True,
) -> Tuple[pd.DataFrame, Dict, np.ndarray, np.ndarray]:
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
        cfg = sample_hyperparams(rng)
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
            label_col=label_col,
            use_upsampling=use_upsampling,
            use_class_weights=use_class_weights,
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
# Single-split training for cascade
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
) -> np.ndarray:
    """
    Train on train_df, predict labels on dev_df.
    Returns predicted labels (numpy array) for dev_df in order.
    """
    if use_upsampling and use_class_weights:
        raise ValueError("Choose either upsampling OR class-weighted loss, not both.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenize_batch = tokenize_batch_factory(transcript_col, tokenizer, max_len)

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

    train_ds = Dataset.from_pandas(train_df_local)
    dev_ds = Dataset.from_pandas(dev_df_local)

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

    num_labels = train_df_local[label_col].nunique()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir="bert_tmp_cascade_single",
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

    # Predict on dev set
    preds_output = trainer.predict(dev_tok)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    return y_pred


def extract_hparams_from_best_cfg(best_cfg: Dict) -> Dict:
    keys = ["learning_rate", "batch_size", "num_train_epochs",
            "max_len", "weight_decay", "warmup_ratio"]
    return {k: best_cfg[k] for k in keys}


def run_cascade_experiment(
    base_df: pd.DataFrame,
    transcript_col: str,
    best_cfg_hc_nonhc: Dict,
    best_cfg_mci_dem: Dict,
    seed: int = 42,
    model_name: str = MODEL_NAME,
    use_class_weights: bool = True,
    use_upsampling: bool = False,
) -> None:
    """
    Train two binary models with best hyperparams on a single
    stratified train/test split, then evaluate the cascaded 3-class predictions.
    """
    print(f"\nRunning cascaded HC -> Non-HC -> (MCI vs Dementia) experiment for {transcript_col}")
    print(f"Backbone model: {model_name}")

    # Original 3-class labels
    y_all = base_df["Label_3cls"].values

    train_idx, test_idx = train_test_split(
        np.arange(len(base_df)),
        test_size=0.2,
        random_state=seed,
        stratify=y_all,
    )

    train_df = base_df.iloc[train_idx].reset_index(drop=True)
    test_df = base_df.iloc[test_idx].reset_index(drop=True)

    # --- Model A: HC vs Non-HC ---
    train_A = train_df.copy()
    test_A = test_df.copy()

    train_A["Label_bin"] = (train_A["Class"] != "HC").astype(int)
    test_A["Label_bin"] = (test_A["Class"] != "HC").astype(int)

    print("\nTraining Model A: HC vs Non-HC on train split")
    hparams_A = extract_hparams_from_best_cfg(best_cfg_hc_nonhc)
    preds_A_test = train_binary_single_split(
        train_df=train_A,
        dev_df=test_A,
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
        seed=seed,
    )

    # --- Model B: MCI vs Dementia ---
    train_B = train_df[train_df["Class"] != "HC"].copy()
    test_B_full = test_df.copy()  # we will still run B on all test samples

    train_B["Label_bin"] = (train_B["Class"] == "Dementia").astype(int)
    # For test, we only need labels for Non-HC if we want diagnostics;
    # but for prediction, we care about all rows, so set any HC label to 0 (won't be used).
    test_B_full["Label_bin"] = np.where(
        test_B_full["Class"] == "Dementia",
        1,
        0,
    )

    print("\nTraining Model B: MCI vs Dementia on Non-HC subset of train")
    hparams_B = extract_hparams_from_best_cfg(best_cfg_mci_dem)
    preds_B_test_full = train_binary_single_split(
        train_df=train_B,
        dev_df=test_B_full,
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
        seed=seed + 1,
    )

    # --- Build cascaded 3-class predictions on test set ---
    # Original 3-class ground truth for test
    y_true_3cls = test_df["Label_3cls"].values

    # Map: 0 -> HC, 1 -> MCI, 2 -> Dementia
    cascade_preds_3cls = []

    for i in range(len(test_df)):
        pred_A = preds_A_test[i]
        if pred_A == 0:
            # Model A says HC
            cascade_preds_3cls.append(0)
        else:
            # Model A says Non-HC -> use Model B
            pred_B = preds_B_test_full[i]  # 0 = MCI, 1 = Dementia
            if pred_B == 0:
                cascade_preds_3cls.append(1)  # MCI
            else:
                cascade_preds_3cls.append(2)  # Dementia

    cascade_preds_3cls = np.array(cascade_preds_3cls)

    # --- Evaluate 3-class cascade ---
    metrics_3cls = compute_metrics_from_labels(y_true_3cls, cascade_preds_3cls)
    print("\nCascaded 3-class metrics on test split:")
    print("  accuracy:", metrics_3cls["accuracy"])
    print("  macro_f1:", metrics_3cls["macro_f1"])
    print("  weighted_f1:", metrics_3cls["weighted_f1"])

    cm = confusion_matrix(y_true_3cls, cascade_preds_3cls, labels=[0, 1, 2])

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Cascaded RoBERTa-large Confusion Matrix - {transcript_col}")
    fig.savefig(f"bert_cascade_confusion_matrix_{transcript_col}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        y_true_3cls,
        cascade_preds_3cls,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print("\nCascaded 3-class classification report:")
    print(report)

    with open(f"bert_cascade_classification_report_{transcript_col}.txt", "w") as f:
        f.write(f"Cascaded 3-class report for {transcript_col} (model={model_name})\n\n")
        f.write(report)
        f.write("\n")
        f.write("Metrics on test split:\n")
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
        print(f"Running cascaded experiments for {transcript_col}")
        print(f"Backbone model: {MODEL_NAME}")
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
            n_trials=N_TRIALS,
            n_splits=N_SPLITS,
            seed=SEED,
            results_csv_path=f"bert_random_search_HC_vs_NonHC_{transcript_col}.csv",
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
        ax.set_title(f"Confusion Matrix - HC vs Non-HC - {transcript_col} (model={MODEL_NAME})")
        fig.savefig(f"bert_confusion_matrix_HC_vs_NonHC_{transcript_col}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        report_A = classification_report(
            y_true_A,
            y_pred_A,
            target_names=["HC", "Non-HC"],
            zero_division=0,
        )
        with open(f"bert_classification_report_HC_vs_NonHC_{transcript_col}.txt", "w") as f:
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
            n_trials=N_TRIALS,
            n_splits=N_SPLITS,
            seed=SEED,
            results_csv_path=f"bert_random_search_MCI_vs_Dem_{transcript_col}.csv",
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
        ax.set_title(f"Confusion Matrix - MCI vs Dementia - {transcript_col} (model={MODEL_NAME})")
        fig.savefig(f"bert_confusion_matrix_MCI_vs_Dem_{transcript_col}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        report_B = classification_report(
            y_true_B,
            y_pred_B,
            target_names=["MCI", "Dementia"],
            zero_division=0,
        )
        with open(f"bert_classification_report_MCI_vs_Dem_{transcript_col}.txt", "w") as f:
            f.write(f"Binary MCI vs Dementia report for {transcript_col} (model={MODEL_NAME})\n\n")
            f.write(report_B)
            f.write("\n")

        # -------------------
        # Cascaded 3-class experiment using best configs
        # -------------------
        run_cascade_experiment(
            base_df=base_df,
            transcript_col=transcript_col,
            best_cfg_hc_nonhc=best_cfg_A,
            best_cfg_mci_dem=best_cfg_B,
            seed=SEED,
            model_name=MODEL_NAME,
            use_class_weights=USE_CLASS_WEIGHTS,
            use_upsampling=USE_UPSAMPLING,
        )

    print("\nAll cascaded experiments finished for PFT / CTD / SFT with RoBERTa-large.")
