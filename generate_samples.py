import pandas as pd
import joblib
import os

# Load the trained model and TF-IDF vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# 10 sample SMS messages
messages = [
    "Congratulations! You have won a free lottery prize. Call now!",
    "Hey, are we meeting for lunch today?",
    "URGENT! You have won a cash prize. Claim it now.",
    "Can you send me the notes from today's class?",
    "Free entry in a weekly competition. Text WIN to 80086.",
    "I will reach home by 7 pm.",
    "You have been selected for a special reward. Claim it now.",
    "Please call me when you are free.",
    "Win a brand new phone by participating in our contest!",
    "Don't forget to bring your assignment tomorrow."
]

# Convert messages into TF-IDF features
X = vectorizer.transform(messages)

# Predict labels
predictions = model.predict(X)

# Get prediction probabilities
probabilities = model.predict_proba(X)

# Find the position of the spam class
spam_index = list(model.classes_).index("spam")

# Store results
results = []

for message, prediction, probability in zip(
    messages,
    predictions,
    probabilities
):
    confidence = max(probability)

    results.append({
        "Message": message,
        "Predicted Label": prediction,
        "Confidence": round(confidence, 4)
    })

# Create DataFrame
df = pd.DataFrame(results)

# Make sure output folder exists
os.makedirs("output", exist_ok=True)

# Save predictions
df.to_csv("output/sample_predictions.csv", index=False)

# Display results
print("\n10 SAMPLE PREDICTIONS")
print("=====================")
print(df.to_string(index=False))

print("\nSample predictions saved successfully!")
print("File: output/sample_predictions.csv")