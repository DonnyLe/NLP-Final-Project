from typing import Dict, Optional, Iterable, List
import itertools
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers, optimizers
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
from transcript_preprocessing import get_stratified_kfold_splits

def prepare_fold_data(
    df,
    transcript_col: str,
    label_col: str,
    train_idx,
    val_idx,
    max_len: int,
):
    """
    Given a DataFrame and train/val indices for one fold, fit a tokenizer on
    the training texts, convert texts to padded integer sequences, and return
    X_train, y_train, X_val, y_val, tokenizer, and vocab_size.
    """
    # Slice train/val splits
    train_texts = df.iloc[train_idx][transcript_col].astype(str).tolist()
    val_texts   = df.iloc[val_idx][transcript_col].astype(str).tolist()
    
    y_train = df.iloc[train_idx][label_col].values
    y_val   = df.iloc[val_idx][label_col].values
    
    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(train_texts)
    
    # Convert texts to integer sequences
    train_seqs = tokenizer.texts_to_sequences(train_texts)
    val_seqs   = tokenizer.texts_to_sequences(val_texts)
    
    # Pad to fixed max_len
    X_train = pad_sequences(
        train_seqs,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )
    X_val = pad_sequences(
        val_seqs,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )
    
    # +1 because we need a slot for 0 (PAD)
    vocab_size = len(tokenizer.word_index) + 1
    
    return X_train, y_train, X_val, y_val, tokenizer, vocab_size

def build_lstm_model(
    config: Dict,
    vocab_size: int,
    max_len: int,
    num_classes: int = 3,
):
    """Build and compile an LSTM or BiLSTM text classification model (Sequential API).

    Architecture:
        Embedding
        (Bi)LSTM
        Dropout
        optional Dense hidden layer + Dropout
        Dense softmax output

    Args:
        config (dict): Hyperparameter configuration. Expected keys:
            - "model_type" (str): "lstm" or "bilstm".
            - "embedding_dim" (int): Size of word embeddings. Default 100.
            - "lstm_units" (int): Number of LSTM units. Default 64.
            - "dropout" (float): Dropout rate for LSTM output and dense layer. Default 0.3.
            - "recurrent_dropout" (float): Recurrent dropout inside LSTM. Default 0.0.
            - "dense_units" (int or None): Size of optional dense hidden layer
                before the output. If 0 or None, this layer is skipped. Default 64.
            - "learning_rate" (float): Adam learning rate. Default 1e-3.
        vocab_size (int): Size of the vocabulary (including PAD and OOV).
        max_len (int): Maximum sequence length for inputs.
        num_classes (int, optional): Number of target classes. Defaults to 3.

    Returns:
        keras.models.Sequential: A compiled Keras model ready for training with
        sparse_categorical_crossentropy loss.
    """
    model_type = config.get("model_type", "lstm")
    embedding_dim = config.get("embedding_dim", 100)
    lstm_units = config.get("lstm_units", 64)
    dropout = config.get("dropout", 0.3)
    recurrent_dropout = config.get("recurrent_dropout", 0.0)
    dense_units: Optional[int] = config.get("dense_units", 64)
    learning_rate = config.get("learning_rate", 1e-3)

    model = Sequential(name=f"{model_type.upper()}_classifier")
    model.add(layers.Input(shape=(max_len,), name="input_ids"))
    # Embedding
    model.add(
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="embedding",
        )
    )
    # LSTM / BiLSTM
    if model_type.lower() == "bilstm":
        model.add(
            layers.Bidirectional(
                layers.LSTM(
                    lstm_units,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=False,
                    name="lstm",
                ),
                name="bilstm",
            )
        )
    else:
        model.add(
            layers.LSTM(
                lstm_units,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=False,
                name="lstm",
            )
        )
    # Dropout after LSTM
    model.add(layers.Dropout(dropout, name="post_lstm_dropout"))
    # Optional Dense hidden layer + Dropout
    if dense_units is not None and dense_units > 0:
        model.add(layers.Dense(dense_units, activation="relu", name="dense"))
        model.add(layers.Dropout(dropout, name="post_dense_dropout"))
    # Softmax output layer 
    model.add(
        layers.Dense(
            num_classes,
            activation="softmax",
            name="classifier",
        )
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.build(input_shape=(None, max_len))

    return model

def run_cv_for_config(
    transcript_col: str,
    df: pd.DataFrame,
    config: dict,
    max_len: int,
    n_splits: int = 5,
    seed: int = 42,
    batch_size: int = 16,
    max_epochs: int = 30,
):
    """Run stratified k-fold CV for a single hyperparameter config on one transcript type."""
    fold_metrics = []

    # Stratified k-fold over this transcript type
    for fold_idx, train_idx, val_idx in get_stratified_kfold_splits(
        transcript_df=df,
        transcript_col=transcript_col,
        label_col="Label",
        n_splits=n_splits,
        seed=seed,
    ):
        X_train, y_train, X_val, y_val, tokenizer, vocab_size = prepare_fold_data(
            df=df,
            transcript_col=transcript_col,
            label_col="Label",
            train_idx=train_idx,
            val_idx=val_idx,
            max_len=max_len,
        )

        model = build_lstm_model(
            config=config,
            vocab_size=vocab_size,
            max_len=max_len,
            num_classes=3,
        )

        # Class weights to deal with imbalance
        classes = np.unique(y_train)
        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_array)}

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=0,
            )
        ]

        # Train the model
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight,
        )

        # Evaluate on validation fold
        y_prob = model.predict(X_val, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
            y_val, y_pred, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(
            y_val, y_pred, average="weighted", zero_division=0
        )

        fold_metrics.append(
            {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
            }
        )

    # Aggregate evaluation metrics across folds
    summary = {}
    for key in fold_metrics[0].keys():
        values = [fm[key] for fm in fold_metrics]
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"std_{key}"] = float(np.std(values))

    return summary

def build_binary_lstm_model(
    config: Dict,
    vocab_size: int,
    max_len: int,
):
    """Build and compile a binary LSTM or BiLSTM text classification model (Sequential API).

    Architecture:
        Embedding
        (Bi)LSTM
        Dropout
        optional Dense hidden layer + Dropout
        Dense sigmoid output (binary)

    Args:
        config (dict): Hyperparameter configuration. Expected keys:
            - "model_type" (str): "lstm" or "bilstm".
            - "embedding_dim" (int): Size of word embeddings. Default 100.
            - "lstm_units" (int): Number of LSTM units. Default 64.
            - "dropout" (float): Dropout rate for LSTM output and dense layer. Default 0.3.
            - "recurrent_dropout" (float): Recurrent dropout inside LSTM. Default 0.0.
            - "dense_units" (int or None): Size of optional dense hidden layer
                before the output. If 0 or None, this layer is skipped. Default 64.
            - "learning_rate" (float): Adam learning rate. Default 1e-3.
        vocab_size (int): Size of the vocabulary (including PAD and OOV).
        max_len (int): Maximum sequence length for inputs.

    Returns:
        keras.models.Sequential: A compiled Keras model ready for binary
        classification with binary_crossentropy loss.
    """
    model_type = str(config.get("model_type", "lstm"))
    embedding_dim = int(config.get("embedding_dim", 100))
    lstm_units = int(config.get("lstm_units", 64))
    dropout = float(config.get("dropout", 0.3))
    recurrent_dropout = float(config.get("recurrent_dropout", 0.0))
    dense_units_val = config.get("dense_units", 64)
    dense_units: Optional[int] = None if dense_units_val is None else int(dense_units_val)
    learning_rate = float(config.get("learning_rate", 1e-3))

    model = Sequential(name=f"{model_type.upper()}_binary_classifier")

    # Embedding
    model.add(layers.Input(shape=(max_len,), name="input_ids"))
    model.add(
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="embedding",
        )
    )

    # LSTM / BiLSTM
    if model_type.lower() == "bilstm":
        model.add(
            layers.Bidirectional(
                layers.LSTM(
                    lstm_units,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=False,
                    name="lstm",
                ),
                name="bilstm",
            )
        )
    else:
        model.add(
            layers.LSTM(
                lstm_units,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=False,
                name="lstm",
            )
        )

    # Dropout after LSTM
    model.add(layers.Dropout(dropout, name="post_lstm_dropout"))

    # Optional dense hidden layer
    if dense_units is not None and dense_units > 0:
        model.add(layers.Dense(dense_units, activation="relu", name="dense"))
        model.add(layers.Dropout(dropout, name="post_dense_dropout"))

    # Sigmoid output layer
    model.add(
        layers.Dense(
            1,
            activation="sigmoid",
            name="classifier",
        )
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

def make_config_grid(
    model_types: Iterable[str],
    embedding_dims: Iterable[int],
    lstm_units: Iterable[int],
    dropout_rates: Iterable[float],
    recurrent_dropout_rates: Iterable[float],
    dense_units_options: Iterable[int],
    learning_rates: Iterable[float],
) -> List[Dict]:
    """
    Create a list of hyperparameter configuration dicts for (Bi)LSTM grid search.

    Args:
        model_types: Values for "model_type" 
        embedding_dims: Values for "embedding_dim" 
        lstm_units: Values for "lstm_units" 
        dropout_rates: Values for "dropout"
        recurrent_dropout_rates: Values for "recurrent_dropout"
        dense_units_options: Values for "dense_units" 
        learning_rates: Values for "learning_rate"

    Returns:
        List[dict]: Each dict is a config with keys:
            "model_type", "embedding_dim", "lstm_units",
            "dropout", "recurrent_dropout", "dense_units", "learning_rate".
    """
    grid: List[Dict] = []
    for (
        model_type,
        emb_dim,
        units,
        dropout,
        rec_dropout,
        dense_units,
        lr,
    ) in itertools.product(
        model_types,
        embedding_dims,
        lstm_units,
        dropout_rates,
        recurrent_dropout_rates,
        dense_units_options,
        learning_rates,
    ):
        config = {
            "model_type": model_type,
            "embedding_dim": emb_dim,
            "lstm_units": units,
            "dropout": dropout,
            "recurrent_dropout": rec_dropout,
            "dense_units": dense_units,
            "learning_rate": lr,
        }
        grid.append(config)
    return grid