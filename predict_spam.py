import joblib

# Load trained model
model = joblib.load("model/spam_model.pkl")

# Load TF-IDF vectorizer
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

print("Spam Classifier loaded successfully!")

# Enter a message
message = input("\nEnter an SMS message: ")

# Convert message to TF-IDF
message_tfidf = tfidf.transform([message])

# Predict
prediction = model.predict(message_tfidf)

# Display result
if prediction[0] == "spam":
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: HAM")