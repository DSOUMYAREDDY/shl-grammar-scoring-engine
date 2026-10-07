## SHL Grammar Scoring Engine

A machine learning solution for the SHL Hiring Assessment 2026 challenge. The goal is to predict a continuous grammar score from spoken audio samples.

## Approach

The solution combines two types of information from the speech recordings:

### 1. Acoustic Features
Audio was processed using `librosa` to extract:
- MFCC features
- MFCC delta and delta-delta features
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero-crossing rate
- RMS energy statistics
- Spectral contrast
- Chroma features
- Silence ratio
- Audio duration

### 2. Linguistic Features

Speech was transcribed using the `faster-whisper` `tiny.en` model.

From the transcripts, the following features were extracted:
- Word count
- Unique word count
- Average word length
- Vocabulary diversity
- Sentence count
- Average words per sentence
- Repeated word count
- Filler-word count
- Punctuation-based sentence count

### 3. Machine Learning Model

An `ExtraTreesRegressor` was trained using the combined acoustic and linguistic features.

Configuration:
- 500 trees
- Random state: 42
- Multi-core CPU training

The target is a continuous grammar score between 0 and 5.

## Evaluation

The model was evaluated using:

- Root Mean Squared Error (RMSE)
- Pearson Correlation

Validation results on an 80/20 train-validation split:

- RMSE: 0.6558
- Pearson Correlation: 0.8789

The Kaggle public leaderboard score for the submitted combined-feature model was:

- Public Score: 0.6064

## Project Structure

```text
shl-grammar-scoring-engine/
├── README.md
└── notebook385ee5fb7d (1).ipynb
