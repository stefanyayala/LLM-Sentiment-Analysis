# **Sentiment Analysis Project Using OpenAI API and Machine Learning**

## **Overview**
This project performs sentiment analysis on a dataset of social media posts. It uses OpenAI's GPT-3.5 model to classify the sentiment of each text entry as **Positive**, **Negative**, or **Neutral**. Additionally, a **Random Forest classifier** is trained on the sentiment labels for further analysis and prediction. The project respects **OpenAI's API usage limits**, adhering to both **requests per minute (RPM)** and **requests per day (RPD)** limits by implementing rate limiting with a sleep interval between API calls.

The dataset is preprocessed to clean and filter it, ensuring only valid sentiment labels are retained. Sentiment predictions are generated using OpenAI’s API and subsequently used to train a Random Forest model. The model’s performance is evaluated using **accuracy** and **F1 score** metrics. The final model, vectorizer, and results are saved for future use.

## **Features**
- Downloads and preprocesses a social media sentiment dataset.
- Integrates OpenAI GPT API to classify text into **Positive**, **Negative**, or **Neutral** sentiment categories.
- Uses **Random Forest** machine learning to train a model based on the sentiment data.
- Implements **rate limiting** to respect API usage constraints (RPM, RPD).
- Saves the results, trained model, and vectorizer for future use.

---

## **Dataset**
The dataset is downloaded using the Kaggle API with the `kagglehub` library. Ensure a valid Kaggle API key (`kaggle.json`) is configured on your system.

### **Dataset Structure**
- **Text**: The social media post text.
- **Sentiment**: The categorized sentiment labels (**Positive**, **Negative**, **Neutral**).

---

## **OpenAI API Configuration**
Ensure you have an OpenAI API key set up as an environment variable (`OPENAI_API_KEY`). The project uses the `gpt-3.5-turbo` model for sentiment analysis.

### **API Rate Limits**
The script respects the following API rate limits:
- **Requests Per Minute (RPM)**: 3
- **Requests Per Day (RPD)**: 200

To respect these limits, the script includes a **20-second delay** between requests. Additionally, it halts execution once the RPD limit is reached.

---

## **Project Workflow**

### 1. **Download Dataset**
The dataset is retrieved from Kaggle using the `kagglehub` library.

```python
path = kagglehub.dataset_download("kashishparmar02/social-media-sentiments-analysis-dataset")
```

### 2. **Preprocess Dataset**
- Load the dataset into a **Pandas DataFrame**.
- Trim spaces from the **Sentiment** column.
- Filter rows where the **Sentiment** is in [Positive, Negative, Neutral].

### 3. **OpenAI Sentiment Analysis**
The function `get_openai_sentiment()`:
- Calls OpenAI API with a prompt to classify sentiments into **Positive**, **Negative**, or **Neutral**.
- Implements rate limiting (20-second delay between requests).

```python
def get_openai_sentiment(text, rpm_limit=3, rpd_limit=200):
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
```

### 4. **Encode Sentiments**
The sentiments returned by OpenAI API are mapped to numerical labels:
- **Negative**: 0
- **Positive**: 1
- **Neutral**: 2

### 5. **Train/Test Split**
The dataset is split into training and testing sets (80/20 split).

### 6. **Text Vectorization**
**CountVectorizer** is used to transform text into feature vectors for model training.

### 7. **Train Random Forest Model**
A **Random Forest** model is trained on the encoded sentiment labels. The model is saved for future use.

### 8. **Evaluate Model**
The trained model is evaluated using **accuracy**, **F1 score**, and a **classification report**.

### 9. **Save Results**
- Processed dataset with predicted sentiments saved as `sentiment_results.csv`.
- The trained model and vectorizer saved as `.pkl` files.

---

## **Files and Outputs**

### **Code Files**
- `sentiment_analysis.py`: The main script for the project.

### **Outputs**
- **Processed Dataset**: `sentiment_results.csv`
- **Trained Model**: `sentiment_model.pkl`
- **Vectorizer**: `vectorizer.pkl`

---

## **Prerequisites**

### **Libraries**
Install the following Python libraries:

```bash
pip install openai pandas scikit-learn joblib kagglehub
```

### **Kaggle API**
Ensure `kaggle.json` is properly configured in your system.

### **OpenAI API**
Set up your OpenAI API key:

```bash
export OPENAI_API_KEY="<your-api-key>"
```

---

## **How to Run**

1. **Download the dataset**:

```python
path = kagglehub.dataset_download("kashishparmar02/social-media-sentiments-analysis-dataset")
```

2. **Run the script**:

```bash
python sentiment_analysis.py
```

3. **Review the results**:
   - Sentiments in `sentiment_results.csv`
   - Trained model in `sentiment_model.pkl`

---

## **Limitations**
- The OpenAI API rate limits (RPM, RPD) restrict the dataset size processed daily.
- Long texts may exceed token limits, requiring truncation or summarization.
- Sentiment analysis results depend on the model’s understanding and may not always align with human interpretation.

---

## **Future Improvements**
- Implement **parallel processing** to handle larger datasets efficiently.
- Fine-tune the **OpenAI model** for custom sentiment classification.
- Explore alternative models for faster inference (e.g., local models or embeddings).

---

## **License**
This project is for educational purposes only. Please adhere to OpenAI's usage guidelines and Kaggle’s dataset terms.
