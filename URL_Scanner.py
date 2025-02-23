from flask import Flask, jsonify
import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

def calculate_entropy(url):
    freq = Counter(url)
    length = len(url)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())

def tokenize_url(url):
    tokens = url.replace('http://', '').replace('https://', '').split('/')
    final_tokens = []
    for token in tokens:
        sub_tokens = token.replace('-', '.').split('.')
        final_tokens.extend(sub_tokens)
    return list(set(final_tokens))

def train_model():
    data_path = 'data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found.")
    
    dataset = pd.read_csv(data_path, delimiter=',', on_bad_lines='skip')
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    labels = dataset.iloc[:, 1].values
    urls = dataset.iloc[:, 0].values
    
    vectorizer = TfidfVectorizer(tokenizer=tokenize_url, token_pattern=None)
    X = vectorizer.fit_transform(urls)
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(f'Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%')
    
    return vectorizer, model

@app.route('/<path:url>', methods=['GET'])
def index(url):
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    transformed_url = vectorizer.transform([url])
    prediction = model.predict(transformed_url)
    entropy_value = calculate_entropy(url)
    
    return jsonify({
        "URL": url,
        "Prediction": "Malicious" if prediction[0] == 'bad' else "Safe",
        "Entropy Score": entropy_value
    })

if __name__ == "__main__":
    vectorizer, model = train_model()
    app.run(debug=True)
