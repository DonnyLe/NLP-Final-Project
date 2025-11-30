import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Regex patterns for transcript cleaning
SPEAKER_PATTERN = re.compile(r"^\s*(Pat|Oth)\s*:", re.IGNORECASE)
PAUSE_PATTERN = re.compile(r"\((\d+)\s*second[s]?\)", re.IGNORECASE)

# Patterns to keep and map to tokens
KEEP_NONVERBAL_PATTERNS: List[Tuple[str, str]] = [
    # laughter
    (r"\(laughs?\)", "<LAUGH>"),
    (r"\(laughter\)", "<LAUGH>"),
    (r"\(oth laughs\)", "<LAUGH>"),
    # sighs
    (r"\(sighs\)", "<SIGH>"),
    # tongue-click / tut
    (r"\(tuts\)", "<TUT>"),
    (r"\(clicking tongue\)", "<CLICK_TONGUE>"),
    (r"\(clicks tongue\)", "<CLICK_TONGUE>"),
]
# Patterns to delete (not useful for cognitive status)
REMOVE_NONVERBAL_PATTERNS: List[str] = [
    # throat / cough / sniff
    r"\(clears throat\)",
    r"\(coughs\)",
    r"\(oth coughs\)",
    r"\(sniffs\)",
    # whistling
    r"\(whistling\)",
    r"\(whistles through teeth\)",
    # environmental / device noises
    r"\(buzzer sounds\)",
    r"\(mobile phone alert\)",
    r"\(phone\s*ringing\)",
    r"\(phone stops\s*ringing\)",
    r"\(alert sound on computer\)",
    r"\(mouse click\)",
    r"\(dog barking\)",
    r"\(noise\)",
    # speech style / name information
    r"\(whispering\)",
    r"\(int2 name\)",
    r"\(oth name\)",
    # unclear
    r"\(\?\)",
]


def extract_patient_speech(text: str, patient_tag: str = "Pat") -> str:
    """
    Extract all speaker indication (i.e. "Pat:") from transcripts and remove any lines not spoken by the patient ("Oth: ...").
    "Pat" indicates patient, "Oth" indicated other speaker.
    Lines with no explicit speaker indication are attributed to the most recent speaker.

    Args:
        text (str): Transcript text.
        patient_tag (str): Tag indicating patient speech.

    Returns:
        str: Cleaned transcript containing only patient speech.
    """
    if not isinstance(text, str):
        return ""

    lines = text.splitlines()
    current_speaker = None
    patient_chunks: List[str] = []
    patient_prefix = patient_tag.lower()[:3] # 'pat

    for raw_line in lines:
        line = raw_line.rstrip()
        speaker = SPEAKER_PATTERN.match(line)
        if speaker:
            current_speaker = speaker.group(1).lower()  # 'pat' or 'oth'
            content = line[speaker.end():].strip()
        else:
            # No explicit speaker label, so attribute to recent speaker
            content = line.strip()

        if current_speaker and current_speaker.startswith(patient_prefix):
            if content:
                patient_chunks.append(content)

    patient_text = " ".join(patient_chunks)
    patient_text = re.sub(r"\s+", " ", patient_text).strip()
    return patient_text

    
def replace_pauses(text: str) -> str:
    """
    Replace occurrences of "(X second(s))" with explicit pause tokens.

    Args:
        text (str): Transcript text.

    Returns:
        str: Transcript text with pause tokens.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")

    def get_token(match: re.Match) -> str:
        seconds = int(match.group(1))
        if seconds <= 2:
            token = "<PAUSE_SHORT>"
        elif seconds <= 5:
            token = "<PAUSE_MEDIUM>"
        else:
            token = "<PAUSE_LONG>"
        return f" {token} "

    text = PAUSE_PATTERN.sub(get_token, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_nonverbals(text: str) -> str:
    """
    Normalize nonverbal markers in the transcript text by replacing some as tokens and removing others.
    Args:
        text (str): Transcript text.
    Returns:
        str: Transcript text with normalized nonverbal markers.
    """
    if not isinstance(text, str):
        return text

    for pattern, token in KEEP_NONVERBAL_PATTERNS:
        text = re.sub(pattern, f" {token} ", text, flags=re.IGNORECASE)

    for pattern in REMOVE_NONVERBAL_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_transcript(raw_text: str) -> str:
    """
    Full preprocessing pipeline for a single transcript field:
    1. Extract patient-only speech (Pat: ...).
    2. Replace (X seconds) with pause tokens.
    3. Normalize nonverbal markers.
    4. Lowercase and collapse whitespace.
    Args:
        raw_text (str): Raw transcript text.
    Returns:
        str: Fully preprocessed transcript text.
    """
    text = extract_patient_speech(raw_text)
    text = replace_pauses(text)
    text = normalize_nonverbals(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_dataframe(
    input_csv_path: str,
    transcript_cols: List[str] = ["Transcript_PFT", "Transcript_CTD", "Transcript_SFT"],
    output_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the original dataset csv, preprocess transcript columns, and
    return a new dataframe.

    This function:
    - Loads the raw CSV.
    - Applies the full preprocessing pipeline to each transcript column
      (patient-only speech, pause tokens, nonverbal normalization, lowercase).
    - Builds a smaller DataFrame containing only Record-ID, Class, and
      the specified transcript columns.
    - Optionally saves this modeling subset to a new CSV.

    Args:
        input_csv_path (str): Path to the raw CSV file.
        transcript_cols (list[str]): Names of transcript columns
        output_csv_path (str, optional): If provided, the cleaned modeling
            subset will be written to this path as a CSV.

    Returns:
        pd.DataFrame: A DataFrame named transcript_df, which
        contains only the columns:
        ["Record-ID", "Class"] + transcript_cols, with all transcript
        fields fully preprocessed and ready for tokenization.
    """

    df_full = pd.read_csv(input_csv_path)

    # Apply preprocessing to each transcript column
    for col in transcript_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].apply(preprocess_transcript)
        else:
            raise ValueError(f"Column '{col}' not found in CSV.")

    required_cols = ["Record-ID", "Class"] + transcript_cols

    transcript_df = df_full[required_cols].copy()

    # add numeric label column for classes
    label_map = {"HC": 0, "MCI": 1, "Dementia": 2}
    transcript_df["Label"] = transcript_df["Class"].map(label_map)

    # Optionally save as CSV
    if output_csv_path is not None:
        transcript_df.to_csv(output_csv_path, index=False)

    return transcript_df

def get_stratified_kfold_splits(
    transcript_df: pd.DataFrame,
    transcript_col: str,
    label_col: str = "Class",
    n_splits: int = 5,
    seed: int = 42,
):
    """
    Generate stratified k-fold train/test splits for a transcript column, preserving 
    the class distribution of the dementia labels.

    Args:
        transcript_df (pd.DataFrame): DataFrame containing at least the transcript column and label column.
        transcript_col (str): Name of the transcript column to use as the input features.
        label_col (str, optional): Column containing class labels
            (default: "Class"). Labels should already be numeric (0/1/2).
        n_splits (int, optional): Number of folds for cross-validation.
            Defaults to 5, which corresponds to ≈80/20 train-test splits.
        seed (int, optional): Random seed used for shuffling and ensuring
            reproducible splits. Defaults to 42.

    Yields:
        tuple: A tuple of the form:
            (fold_idx, train_idx, test_idx)
        where:
            fold_idx (int): The 1-based fold number.
            train_idx (np.ndarray): Indices of training samples for this fold.
            test_idx (np.ndarray): Indices of test samples for this fold.

    Raises:
        ValueError: If transcript_col or label_col are not present in the
            supplied DataFrame.
    """
    if transcript_col not in transcript_df.columns:
        raise ValueError(f"Transcript column '{transcript_col}' not found in transcript_df.")
    if label_col not in transcript_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in transcript_df.")

    X = transcript_df[transcript_col].values
    y = transcript_df[label_col].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        yield fold_idx, train_idx, test_idx

def load_transcript_splits(
        csv_path: str, 
        transcript_cols: List[str] = ["Transcript_PFT", "Transcript_CTD", "Transcript_SFT"]
    ) -> dict:
    """
    Load cleaned transcripts CSV, drop NaNs per transcript type,
    and return a dict mapping transcript column name to its cleaned DataFrame.
    """
    transcript_df = pd.read_csv(csv_path)
    df_by_transcript = {}

    for col in transcript_cols:
        # Drop NaNs only in this transcript type column
        df_clean = transcript_df.dropna(subset=[col]).copy()
        df_by_transcript[col] = df_clean

    return df_by_transcript


if __name__ == "__main__":
    INPUT_CSV = "data/dementia_data.csv"
    OUTPUT_CSV = "data/transcripts_cleaned.csv"

    transcript_df = preprocess_dataframe(
        input_csv_path=INPUT_CSV,
        transcript_cols=["Transcript_PFT", "Transcript_CTD", "Transcript_SFT"],
        output_csv_path=OUTPUT_CSV,
    )

    print("transcript_df shape:", transcript_df.shape)
    print(transcript_df.head())

    # 2. Print out token sizes for cleaned transcripts
    for col in ["Transcript_PFT", "Transcript_CTD", "Transcript_SFT"]:
        if col in transcript_df.columns:
            lengths = (
                transcript_df[col]
                .fillna("")
                .str.split()
                .apply(len)
            )
            print(f"{col}:")
            print(f"  max length (tokens) = {lengths.max()}")
            print(f"  mean length (tokens) = {lengths.mean():.2f}")
            print(f"  95th percentile = {lengths.quantile(0.95):.0f}")
            print()

    for fold_idx, train_idx, test_idx in get_stratified_kfold_splits(
        transcript_df, transcript_col="Transcript_PFT", label_col="Class", n_splits=5):
        print(f"Fold {fold_idx}: ")
        print(f"train size = {len(train_idx)}, test size = {len(test_idx)}")

    
