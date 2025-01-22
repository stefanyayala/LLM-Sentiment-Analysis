import os
import openai
import pandas as pd
import json
import kagglehub
import time
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Secure API Key Management
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize counter for API calls
call_count = 0


def get_openai_sentiment(text, rpm_limit=3, rpd_limit=200):
    """
    Calls the OpenAI API while respecting RPM and RPD limits.
    Args:
        text (str): The text to analyze.
        rpm_limit (int): Requests per minute limit (default 3).
        rpd_limit (int): Requests per day limit (default 200).
    Returns:
        str: Sentiment label from OpenAI API or "Unknown" if not valid.
    """
    global call_count
    if call_count >= rpd_limit:
        print("RPD limit reached. Halting execution.")
        return "Error"

    # Respect RPM limit
    time.sleep(20)  # Sleep for 20 seconds to respect the RPM limit of 3 per minute

    # Update counter for API calls
    call_count += 1

    try:
        # Correct the prompt to be a string
        prompt = f"Analyze the sentiment of the following text and classify as Positive, Negative, or Neutral:\n\nText: {text}\nSentiment:"
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",  # Adjust engine according to your needs
            prompt=prompt,
            max_tokens=50,
            temperature=0.5,
        )

        sentiment = response['choices'][0]['text'].strip()

        # Ensure the sentiment is valid
        if sentiment in ['Positive', 'Negative', 'Neutral']:
            return sentiment
        return "Unknown"

    except Exception as e:
        print(f"Error: {e}")
        return "Error"


# Download dataset using Kaggle API (ensure kaggle.json is properly set up)
path = kagglehub.dataset_download("kashishparmar02/social-media-sentiments-analysis-dataset")

# Locate CSV file in the downloaded path
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break
else:
    raise FileNotFoundError("No CSV file found in the dataset folder.")

# Load and preprocess dataset
data = pd.read_csv(csv_file)
print("Dataset loaded successfully.")
print(f"Initial shape: {data.shape}")

# Trim spaces from the 'Sentiment' column
data['Sentiment'] = data['Sentiment'].str.strip()

# Keep only the 'Text' and 'Sentiment' columns
data = data[['Text', 'Sentiment']]

# Filter rows where 'Sentiment' is in ['Positive', 'Negative', 'Neutral']
data = data[data['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

print("Filtered dataset:")
print(data.shape)
print(data.head())

# Apply OpenAI sentiment analysis to the 'Text' column
data['Sentiment'] = data['Text'].apply(get_openai_sentiment)

# Encode labels
data['label'] = data['Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data['Text'], data['label'], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print("Accuracy:", accuracy)
print("F1 Score (Macro):", f1)
print("Classification Report:\n",
      classification_report(y_test, y_pred, target_names=['Negative', 'Positive', 'Neutral']))

# Save the results to a new CSV file
data['predicted_label'] = model.predict(vectorizer.transform(data['Text']))
data.to_csv("sentiment_results.csv", index=False)

# Save the model and vectorizer for reuse
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")
