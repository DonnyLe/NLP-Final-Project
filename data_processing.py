import pandas as pd

# load csv files to data frame -> take transcripts from clean_train_transcripts.csv and other columns from dementia_data.csv
DEMENTIA_DATA = 'data/dementia_data.csv'
TRAIN_TRANSCRIPTS = 'data/clean_train_transcripts.csv'
DEMENTIA_COLUMNS = [
    'Record-ID',
    'TrainOrDev',
    'Class',  #  0 (Healthy Control), 1 (Mild Cognitive Impairment), or 2 (Dementia)
    'Gender',
    'Age',
    'Converted-MMSE',
    # precomputed linguistic features
    'filler_count',
    'token_count',
    'type_count',
    'type_token_ratio',
    'ma_ttr',
    'brunets_index',
    'content_density',
    'repetitions',
    'sentence_count',
    'average_words_per_sentence'
]

# TRAIN_TRANSCRIPTS contains already cleaned transcripts
TRANSCRIPT_COLUMNS = ['Record-ID','Transcript_CTD','Transcript_PFT','Transcript_SFT']

# Load CSVs
dementia_df = pd.read_csv(DEMENTIA_DATA)
transcripts_df = pd.read_csv(TRAIN_TRANSCRIPTS)

dementia_df = dementia_df[DEMENTIA_COLUMNS]
transcripts_df = transcripts_df[TRANSCRIPT_COLUMNS]

# Merge dataframes on Record-ID
combined_df = pd.merge(
    dementia_df,
    transcripts_df,
    on='Record-ID',
    how='inner'
)

print("Combined df shape: ", combined_df.shape)

# Split into train and test sets
train_df = combined_df[combined_df['TrainOrDev'] == 'train'].reset_index(drop=True)
test_df  = combined_df[combined_df['TrainOrDev'] == 'dev'].reset_index(drop=True)

print("Train df shape:", train_df.shape)
print("Test df shape:", test_df.shape)

# Vocabulary richness features

VOCAB_FEATURE_COLS = [
    'Record-ID',
    'token_count',
    'type_count',
    'type_token_ratio',
    'ma_ttr',
    'brunets_index',
    'content_density',
    'repetitions'
]

vocab_richness_train = train_df[VOCAB_FEATURE_COLS].copy()
vocab_richness_test  = test_df[VOCAB_FEATURE_COLS].copy()

print("vocab_richness_train shape:", vocab_richness_train.shape)
print("vocab_richness_test shape:", vocab_richness_test.shape)

# Fluency / discourse features

FLUENCY_FEATURE_COLS = [
    'Record-ID',
    'filler_count',
    'sentence_count',
    'average_words_per_sentence'
]

fluency_train = train_df[FLUENCY_FEATURE_COLS].copy()
fluency_test  = test_df[FLUENCY_FEATURE_COLS].copy()

print("fluency_train shape:", fluency_train.shape)
print("fluency_test shape:", fluency_test.shape)

vocab_richness_train.head()
fluency_train.head()
