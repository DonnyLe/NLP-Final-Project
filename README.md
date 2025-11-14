### NLP Final Project - Dementia Classifier
Overview
This dataset contains linguistic, acoustic, and demographic information extracted from spoken tasks
performed by participants with varying cognitive statuses. It is intended for machine learning, linguistic
analysis, and dementia-related research.

File Description
dementia_data.csv contains one row per participant/task session including metadata, transcripts, and
pre-computed speech and language features.

1. Record-ID

Type: string

Description: Unique ID for each participant/session.

Examples: 001, A03

Range: Any alphanumeric string.

2. TrainOrDev

Type: string

Description: Indicates whether the sample belongs to the training or development set.

Examples: Train, Dev

Range: {Train, Dev}

3. Class

Type: string

Description: Diagnostic group of the participant.

Examples: Control, MCI, Dementia

Range: Finite discrete set.

4. Gender

Type: integer

Description: Encoded gender

0 = female

1 = male

Examples: 0, 1

Range: {0, 1}

5. Age

Type: string (may include ranges or approximations)

Description: Participant age.

Examples: 78, 70–75, ~80

Range: approx. 50–95

6. Converted-MMSE

Type: float

Description: Cognitive score (Mini-Mental State Examination), converted to a 0–30 scale.

Examples: 22.0, 28.5, 15.0

Range: 0–30

7. Transcript_PFT

Type: string

Description: Transcript of the Picture Description Task (PFT).

Examples: Full paragraph describing a scene.

Range: Variable-length text.

8. Transcript_CTD

Type: string

Description: Transcript of the Category/Topic Description Task.

Examples: Multi-sentence descriptive responses.

Range: Variable-length text.

Notes: One missing value.

9. Transcript_SFT

Type: string

Description: Semantic Fluency Task transcript (e.g., listing items).

Examples: “dog, cat, horse”

Range: Short lists or sequences of words.

10. Class_label

Type: integer

Description: Numeric encoding of cognitive diagnosis.

Examples: 0, 1, 2

Range: Small set of class integers.

11. total_seconds

Type: integer

Description: Length of the recording in seconds.

Examples: 60, 180, 240

Range: ~10–600

12. Parentheses_Content

Type: string

Description: Extracted content inside parentheses from transcripts.

Examples: "(laughs)", "(pause)"

Range: Short strings or empty.

13. CTD_Cleaned

Type: string

Description: Cleaned and normalized version of the CTD transcript.

Examples: Lowercased, punctuation-stripped text.

Range: Variable-length cleaned text.

14. found_fillers

Type: string

Description: Serialized list of detected filler words.

Examples: "['um', 'uh']", "[]"

Range: 0 or more items.

15. filler_list

Type: string

Description: Ordered list of filler word occurrences.

Examples: "['like', 'um']", "['uh']"

Range: 0–many items.

16. filler_count

Type: integer

Description: Total number of filler words detected.

Examples: 0, 3, 12

Range: 0–20

17. token_count

Type: integer

Description: Total number of tokens (words) in the transcript.

Examples: 50, 100, 250

Range: 0–400

18. type_count

Type: integer

Description: Number of unique word types.

Examples: 20, 70, 120

Range: 0–200

19. type_token_ratio

Type: float

Description: Lexical diversity = type_count / token_count.

Examples: 0.20, 0.55

Range: 0–1

20. ma_ttr

Type: float

Description: Moving-Average Type Token Ratio (stable lexical diversity measure).

Examples: 0.45, 0.60

Range: typically 0–1

21. brunets_index

Type: float

Description: Brunet’s Index, a lexical richness measure.

Examples: 10.2, 12.5

Range: typically 8–18

22. content_density

Type: float

Description: Ratio of content words (nouns, verbs, adjectives, adverbs) to total words.

Examples: 0.45, 0.62

Range: 0–1

23. repetitions

Type: dict

Description: Dictionary of repeated words and their frequencies.

Examples: {}, {"the": 10, "is": 5}

Range: Arbitrary dictionary of word → count pairs

24. sentence_count

Type: integer

Description: Number of sentences in the transcript.

Examples: 0, 4, 7

Range: 0–15

25. average_words_per_sentence

Type: float

Description: Mean sentence length.

Examples: 15.2, 44.0

Range: 0–50


Notes & Recommendations
Age should be cleaned for numeric analysis.
repetitions contains Python dicts requiring parsing.
One missing value exists in Transcript_CTD.
