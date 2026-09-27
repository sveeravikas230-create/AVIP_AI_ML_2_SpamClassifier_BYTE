import os
import pandas as pd

# --------------------------------------------------
# TASK 2: SMS SPAM CLASSIFIER
# Step 1: Load the dataset
# --------------------------------------------------

# Dataset location
data_path = os.path.join("data", "SMSSpamCollection")

# Check whether the dataset exists
if not os.path.exists(data_path):
    print("ERROR: Dataset not found!")
    print("Please make sure the file is located here:")
    print("data/SMSSpamCollection")
    exit()

# Load the dataset
df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# --------------------------------------------------
# Display dataset information
# --------------------------------------------------

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
import os
import pandas as pd

# --------------------------------------------------
# TASK 2: SMS SPAM CLASSIFIER
# Step 1: Load the dataset
# --------------------------------------------------

# Dataset location
data_path = os.path.join("data", "SMSSpamCollection")

# Check whether the dataset exists
if not os.path.exists(data_path):
    print("ERROR: Dataset not found!")
    print("Please make sure the file is located here:")
    print("data/SMSSpamCollection")
    exit()

# Load the dataset
df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# --------------------------------------------------
# Display dataset information
# --------------------------------------------------

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
# --------------------------------------------------
# STEP 3: Split data into training and testing sets
# --------------------------------------------------

from sklearn.model_selection import train_test_split

# Input data
X = df["message"]

# Target/output
y = df["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split completed!")

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print("Training labels:", len(y_train))
print("Testing labels:", len(y_test))
# --------------------------------------------------
# STEP 4: Convert text into TF-IDF features
# --------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

# Learn vocabulary from training messages
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing messages using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF conversion completed!")

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)
# Train Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

# Save the trained model
joblib.dump(model, "model/spam_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\nModel and TF-IDF vectorizer saved successfully!")
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create output folder
os.makedirs("output", exist_ok=True)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()
plt.title("Spam SMS Classifier - Confusion Matrix")

# Save image
plt.savefig("output/confusion_matrix.png", bbox_inches="tight")

plt.show()

print("Confusion matrix saved successfully!")