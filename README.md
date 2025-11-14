### NLP Final Project - Dementia Classifier
Overview
This dataset contains linguistic, acoustic, and demographic information extracted from spoken tasks
performed by participants with varying cognitive statuses. It is intended for machine learning, linguistic
analysis, and dementia-related research.

File Description
dementia_data.csv contains one row per participant/task session including metadata, transcripts, and
pre-computed speech and language features.

Metadata Columns
1. Record-ID
   - Type: string
   - Description: Unique ID for each participant/session.
   - Examples: 001, A03
   - Range: Any alphanumeric string.
2. TrainOrDev
   - Type: string
   - Description: Indicates whether the sample belongs to the training or development set.
   - Examples: Train, Dev
   - Range: {Train, Dev}
3. Class
   - Type: string

Description: Diagnostic group of the participant.

Examples: Control, MCI, Dementia

Range: Finite discrete set.

Gender

Type: integer

Description: Encoded gender

0 = female

1 = male

Examples: 0, 1

Range: {0, 1}

Age

Type: string (may contain ranges or approximate values)

Description: Participant age.

Examples: 78, 70-75, ~80

Range: approx. 50–95.

Converted-MMSE

Type: float

Description: Cognitive score (Mini-Mental State Examination), converted to unified scale.

Examples: 22.0, 28.5, 15.0

Range: 0–30

Transcripts
Transcript_PFT

Type: string

Description: Transcript of the Picture Description Task (PFT).

Examples: Full paragraph describing a picture.

Transcript_CTD

Type: string

Description: Transcript of Category/Topic Description Task.

Examples: multi-sentence descriptive responses.

Notes: Contains one missing value.

Transcript_SFT

Type: string

Description: Semantic Fluency Task transcript
(e.g., listing animals).

Examples: "dog, cat, horse"

🗣 Speech Timing Features
total_seconds

Type: int

Description: Length of the recording in seconds.

Range: ~10–600 seconds.

speaking_rate_wpm

Type: float

Description: Words per minute (WPM).

Range: ~50–180.

pause_count

Type: int

Description: Number of pauses detected in speech.

Range: 0–40.

avg_pause_length

Type: float

Description: Average pause duration in seconds.

Range: 0–2 seconds.

📝 Linguistic Features
num_words

Type: int

Description: Total words spoken.

Range: 0–400.

unique_words

Type: int

Description: Count of unique words spoken.

Range: 0–200.

type_token_ratio

Type: float

Description: unique_words / num_words

Range: 0–1

filler_word_count

Type: int

Description: Filler words (um, uh, like, etc.).

Range: 0–20.

avg_word_length

Type: float

Description: Average number of characters per word.

Range: ~3–6.

total_syllables

Type: int

Description: Total syllables spoken.

Range: 0–1000.

avg_syllables_per_word

Type: float

Description: Mean syllables per word.

Range: 1–2.5.

sentence_count

Type: int

Description: Number of sentences in the transcript.

Range: 0–15.

average_words_per_sentence

Type: float

Description: Mean sentence length.

Range: 0–50.

repetitions

Type: dict

Description: Counts of repeated words in the transcript.

Examples:
{},
{"the": 10, "is": 5}

Range: Arbitrary dictionary of word → count pairs.


Notes & Recommendations
Age should be cleaned for numeric analysis.
repetitions contains Python dicts requiring parsing.
One missing value exists in Transcript_CTD.
