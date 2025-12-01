import re
from typing import Dict
import numpy as np
import pandas as pd


def compute_linguistic_features_for_column(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Given a DataFrame and the name of a transcript column,
    compute a set of linguistically-motivated numeric features 
    mimicing those in the initial dementia dataset and return
    a new DataFrame with those feature columns added.

    The features added are:
      - token_count
      - type_count
      - type_token_ratio
      - brunets_index
      - ma_ttr
      - sentence_count
      - average_words_per_sentence
      - filler_count
      - found_fillers (0/1)
      - content_density
      - repetitions
    """
    out = df.copy()

    texts = out[text_col].fillna("")

    # Basic whitespace tokenization
    token_lists = texts.apply(lambda s: s.split())

    out["token_count"] = token_lists.apply(len)
    out["type_count"] = token_lists.apply(lambda toks: len(set(toks)))
    out["type_token_ratio"] = out["type_count"] / out["token_count"].replace(0, np.nan)

    # Calculate Brunet's index: W = N^(V^-0.165)
    def brunet_index(tokens):
        N = len(tokens)
        V = len(set(tokens))
        if N == 0 or V == 0:
            return np.nan
        return N ** (V ** -0.165)

    out["brunets_index"] = token_lists.apply(brunet_index)

    # Calculate moving-average TTR over windows of 50 tokens
    def ma_ttr(tokens, window: int = 50):
        if not tokens:
            return np.nan
        ttrs = []
        for i in range(0, len(tokens), window):
            chunk = tokens[i : i + window]
            if not chunk:
                continue
            ttrs.append(len(set(chunk)) / len(chunk))
        return float(np.mean(ttrs)) if ttrs else np.nan

    out["ma_ttr"] = token_lists.apply(ma_ttr)

    # Calculate sentence count and average words per sentence 
    def split_sentences(text: str):
        parts = re.split(r"[.!?]+", text)
        return [p for p in parts if p.strip()]

    sentence_lists = texts.apply(split_sentences)
    out["sentence_count"] = sentence_lists.apply(len)
    out["sentence_count"] = out["sentence_count"].replace(0, np.nan)
    out["average_words_per_sentence"] = out["token_count"] / out["sentence_count"]

    FILLERS = {'right', 'actually', 'so', 'uh', 'like', 'basically', 'um', 'i mean', 'well', 'er', 'you know'}
    PAUSE_TOKENS = {"<pause_short>", "<pause_medium>", "<pause_long>"}

    # Calculate presence and count of filler words
    def count_fillers(tokens):
        count = 0
        for w in tokens:
            w_clean = w.strip(".,;:!?\"'()").lower()
            if w_clean in FILLERS:
                count += 1
        return count

    out["filler_count"] = token_lists.apply(count_fillers)
    out["found_fillers"] = (out["filler_count"] > 0).astype(int)

    # Calculate Content density (proportion of tokens that are not stopwords, fillers, or pauses)
    STOPWORDS = {
        "the", "is", "and", "a", "an", "to", "of", "in", "it",
        "that", "this", "on", "for", "with", "as", "at", "by",
        "from", "or", "be",
    }

    def content_density(tokens):
        if not tokens:
            return np.nan
        content = 0
        total = len(tokens)
        for w in tokens:
            if w in PAUSE_TOKENS:
                continue
            w_clean = w.strip(".,;:!?\"'()").lower()
            if w_clean in FILLERS:
                continue
            if w_clean in STOPWORDS:
                continue
            content += 1
        return content / total

    out["content_density"] = token_lists.apply(content_density)

    # Calculate repetitions of tokens
    out["repetitions"] = out["token_count"] - out["type_count"]

    return out
