import pandas as pd
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify

# Download NLTK resources (if not already downloaded, ensure they are available in your environment)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))

# --- Preprocessing functions (same as in the notebook) ---
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in stop_words])

def preprocess_text(text):
    text = remove_html_tags(text)
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    # Tokenization and lemmatization would typically be here, but CountVectorizer handles tokenization
    # and we can skip explicit lemmatization for a simpler app.py if vectorizer was trained without it
    # For this specific setup, we'll just feed the preprocessed string to the vectorizer.
    return text

# --- Load the saved vectorizer and model ---
# Ensure 'vectorizer.pkl' and 'logistic_regression_model.pkl' are in the same directory
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Vectorizer and model loaded successfully.")
except FileNotFoundError:
    print("Error: 'vectorizer.pkl' or 'logistic_regression_model.pkl' not found. Please run the saving cells first.")
    exit()

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API! Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Please provide a JSON object with a "text" field'}), 400

    review_text = request.json['text']

    # Preprocess the input text
    processed_text = preprocess_text(review_text)

    # Vectorize the preprocessed text
    # CountVectorizer expects an iterable, so we pass [processed_text]
    text_vectorized = vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(text_vectorized)

    # Map numerical prediction back to sentiment label
    sentiment_label = 'positive' if prediction[0] == 1 else 'negative'

    return jsonify({
        'input_text': review_text,
        'processed_text': processed_text,
        'prediction': int(prediction[0]), # Convert numpy int to Python int
        'sentiment': sentiment_label
    })

if __name__ == '__main__':
    # You can run this app using 'python app.py'
    # For development, you can use app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
