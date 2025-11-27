from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

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
