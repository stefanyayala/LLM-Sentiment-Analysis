# Sentiment Analysis Project Using OpenAI API and Machine Learning

## Overview
This project analyzes sentiments from a dataset of social media posts using OpenAI’s GPT model for nuanced sentiment classification. Additionally, it employs machine learning techniques to train a classifier based on the generated sentiments. The project adheres to OpenAI's API usage limits and implements rate limiting and batching mechanisms.

## Features
- Downloads and preprocesses a social media sentiment dataset.
- Integrates OpenAI GPT API to classify text into nuanced sentiment categories.
- Uses machine learning (Gradient Boosting) to train a model based on the sentiment data.
- Implements rate limiting to respect API usage constraints (RPM, RPD).
- Saves results and trained models for future use.

---

## Dataset
The dataset is retrieved from Kaggle using the `kagglehub` library. Ensure you have a valid Kaggle API key (`kaggle.json`) configured in your system.

### Dataset Structure
- **Text**: The social media post text.
- **Sentiment**: The categorized sentiment labels (output of OpenAI API).

---

## OpenAI API Configuration
Ensure you have an OpenAI API key configured as an environment variable (`OPENAI_API_KEY`). If not, set it up in your system or directly in the code. The project uses the `gpt-3.5-turbo` model for sentiment analysis.

### API Rate Limits
The following limits are respected in the code:
- **Requests Per Minute (RPM)**: 3
- **Requests Per Day (RPD)**: 200

The script implements `time.sleep(20)` between API calls to respect the RPM limits. A counter ensures the script halts after reaching the daily RPD limit.

---

## Project Workflow

### 1. Download Dataset
The dataset is downloaded from Kaggle using the `kagglehub` library.
```python
path = kagglehub.dataset_download("kashishparmar02/social-media-sentiments-analysis-dataset")
```

### 2. Preprocess Dataset
- Load the dataset into a Pandas DataFrame.
- Apply text cleaning if necessary (e.g., removing special characters, URLs).

### 3. OpenAI Sentiment Analysis
A function `get_openai_sentiment()` is implemented to:
- Call the OpenAI API with a prompt to classify sentiments.
- Implement rate limiting (20 seconds delay between requests).
- Handle errors gracefully.

```python
def get_openai_sentiment(text, rpm_limit=3, rpd_limit=200):
    try:
        prompt = (
            "Analyze the sentiment of the following text and classify it into one or more categories:\n\n"
            "Categories: Positive, Negative, Neutral, Joy, Sadness, Anger, Fear, Surprise, Gratitude, etc.\n\n"
            f"Text: {text}\nSentiment:"
        )
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=50,
            temperature=0.5,
        )
        sentiment = response['choices'][0]['text'].strip().split(", ")
        return sentiment
    except Exception as e:
        print(f"Error: {e}")
        return ["Error"]
```

### 4. Encode Sentiments
The sentiments returned by the OpenAI API are encoded using `MultiLabelBinarizer` for compatibility with machine learning models.

### 5. Train/Test Split
The dataset is split into training and testing sets (80/20 split) for model training and evaluation.

### 6. TF-IDF Vectorization
The text data is transformed into numerical format using `TfidfVectorizer`. The vectorized text is then used to train the model.

### 7. Train Gradient Boosting Model
A Gradient Boosting Classifier is trained on the encoded sentiments. The trained model is saved for future use.

### 8. Evaluate Model
The trained model is evaluated using accuracy, F1 score, and a classification report.

### 9. Save Results
- The processed dataset with encoded sentiments is saved as `sentiment_results.csv`.
- The trained model, vectorizer, and label binarizer are saved as `.pkl` files.

---

## Files and Outputs

### Code Files
- `sentiment_analysis.py`: Main script for the project.

### Outputs
- **Processed Dataset**: `sentiment_results.csv`
- **Trained Model**: `sentiment_model.pkl`
- **TF-IDF Vectorizer**: `vectorizer.pkl`
- **Label Binarizer**: `label_binarizer.pkl`

---

## Prerequisites

### Libraries
Install the following Python libraries:
```bash
pip install openai pandas scikit-learn joblib kagglehub
```

### Kaggle API
Ensure `kaggle.json` is properly configured in your system. Place it in `~/.kaggle/` or the current working directory.

### OpenAI API
Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="<your-api-key>"
```

---

## How to Run
1. Download the dataset:
   ```python
   path = kagglehub.dataset_download("kashishparmar02/social-media-sentiments-analysis-dataset")
   ```

2. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

3. Review the results:
   - Sentiments in `sentiment_results.csv`
   - Trained model in `sentiment_model.pkl`

---

## Limitations
- The OpenAI API rate limits (RPM, RPD) restrict the dataset size processed daily.
- Long texts may exceed token limits, requiring truncation or summarization.
- Sentiment analysis results depend on the model’s understanding and may not always align with human interpretation.

---

## Future Improvements
- Implement parallel processing to handle larger datasets efficiently.
- Use OpenAI fine-tuning for custom sentiment classification.
- Explore alternative models for faster inference (e.g., local models or embeddings).

---

## License
This project is for educational purposes only. Please adhere to OpenAI's usage guidelines and Kaggle’s dataset terms.
