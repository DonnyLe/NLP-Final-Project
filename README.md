# Exploring Dementia Classification Models

This project investigates a variety of models and approaches for classifying cognitive speech transcripts as Healthy Control (HC), Mild Cognitive Impairment (MCI), and Dementia—using three types of speech transcripts: **Phonemic Fluency Task (PFT)**, **Cookie Theft Description (CTD)**, and **Semantic Fluency Task (SFT)**.

Our primary goal is to compare how different modeling approaches perform and to identify which transcript types and architectures provide the most reliable signals for dementia detection.

The [dataset](https://www.kaggle.com/datasets/tahouramorovati/dementia-detection-using-speech) consists of **157 total samples**: 82 HC, 59 MCI, and 16 Dementia

All experiments were evaluated using **stratified 5-fold cross-validation** to maintain consistent label distribution across folds.

---

## Project Goals
- Determine whether linguistic patterns in clinical transcripts can differentiate HC, MCI, and Dementia.
- Determine if models perform better at distinugishing healthy vs. cognitively impaired speech with binary classification by combining MCI and Dementia into one Cognitively Impaired (CI) class.
- Compare multiple model families, including Logistic Regression, LSTMs, and BERT.
- Assess how transcript type (CTD, PFT, SFT) influences classification performance.
- Identify which modeling approaches are most robust under a small, imbalanced dataset.

---

## Models Implemented

### 1. Logistic Regression
A multinomial logistic regression model was trained separately for each transcript type using **TF-IDF features**.  
Key properties:
- Strong classical baseline for sparse text data  
- Evaluated in both multiclass and binary (HC vs. CI) tasks  
- Captures lexical frequency patterns without modeling word order  

---

### 2. Logistic Regression with Feature Engineering
The original dataset included linguistic features for CTD (e.g., token count, type–token ratio, Brunet’s index, filler count, repetitions). We replicated these feature extraction procedures for PFT and SFT to create a unified feature set across all transcript types.

We tested the model with and without the speech transcripts in the input.

This model combines:
- TF-IDF text representation  
- Linguistic / psycholinguistic features  
- Metadata where available  

---

### 3. LSTM / BiLSTM Neural Networks
We implemented both LSTM and BiLSTM sequence models to capture temporal and contextual patterns in each transcript.

Model components:
- Tokenization with `<UNK>` handling for out-of-vocabulary words  
- Padding to fixed sequence lengths per transcript type  
- Embedding layer → (Bi)LSTM → Dropout → optional dense hidden layer → softmax output (or sigmoid for binary)

---

### 4. BERT Fine-Tuning (Transformer Model)
We fine-tuned **bert-base-uncased** separately for CTD, PFT, and SFT to evaluate transformer-based performance.

Setup included:
- HuggingFace `Trainer` API  
- Truncation/padding to a fixed maximum token length  
- Hyperparameter exploration of learning rate, warmup ratio, and epochs  

