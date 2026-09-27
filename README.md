# Spam SMS/Email Classifier

## Project Overview

This project classifies SMS/email messages as either Ham (legitimate)
or Spam using Natural Language Processing and Machine Learning.

## Dataset

The project uses the UCI SMS Spam Collection dataset.

The dataset contains SMS messages labeled as ham or spam.

## Preprocessing

The following preprocessing steps were performed:

1. Loaded the SMS dataset.
2. Removed duplicate messages.
3. Separated labels and messages.
4. Split the dataset into training and testing data.
5. Converted text into numerical features using TF-IDF.
6. Used a maximum of 5,000 TF-IDF features.

## Machine Learning Model

Multinomial Naive Bayes was used as the classification algorithm.

TF-IDF was used to convert the text messages into numerical features.

## Evaluation

The model was evaluated on the test dataset using accuracy, precision,
recall, F1-score, and a confusion matrix.

### Test Set Performance

- Accuracy: 0.9722
- Precision: 1.0000
- Recall: 0.7919
- F1-score: 0.8840

### Confusion Matrix

```text
[[966, 0],
 [31, 118]]

## Saved Model

The trained model is saved as:

model/spam_model.pkl

The TF-IDF vectorizer is saved as:

model/tfidf_vectorizer.pkl

## Sample Predictions

Ten sample SMS messages were tested with the trained model.

The predictions and confidence scores are saved in:

output/sample_predictions.csv

## Confusion Matrix

The confusion matrix is saved as:

output/confusion_matrix.png

## How to Install

Install the required Python packages using:

```bash
pip install -r requirements.txt