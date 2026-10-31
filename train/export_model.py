"""
Rebuild the model from NLP_Algo_Harness.ipynb and export it for the website.

Runs the notebook's exact pipeline on the IMDB 50k reviews (clean -> tokenize ->
stop words -> WordNet lemmatize -> TF-IDF top 5,000 words -> Logistic Regression),
then writes data/model.json, which the landing page loads to score text in the browser.

Usage (Windows, from the repo folder):
    pip install pandas scikit-learn nltk
    python train/export_model.py "path\\to\\IMDB Dataset.csv"
"""
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

for pkg in ("punkt_tab", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

csv_path = sys.argv[1] if len(sys.argv) > 1 else "IMDB Dataset.csv"
df = pd.read_csv(csv_path).rename(columns={"review": "text", "sentiment": "target"})
df["target"] = (df["target"] == "positive").astype(int)

# --- Step 3.2: the notebook's clean_text, unchanged ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
surface_counts = Counter()

def preprocess(text):
    tokens = nltk.word_tokenize(clean_text(text))                 # Step 3.3
    tokens = [w for w in tokens if w not in stop_words]           # Step 3.4
    surface_counts.update(tokens)
    return " ".join(lemmatizer.lemmatize(w) for w in tokens)      # Step 3.5

print("Preprocessing 50,000 reviews...")
df["text"] = df["text"].apply(preprocess)

# --- Step 4: split and TF-IDF, same settings as the notebook ---
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["target"], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Step 5: the harness winner ---
model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_tfidf, y_train)
pred = model.predict(X_test_tfidf)
proba = model.predict_proba(X_test_tfidf)[:, 1]
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc_labels": roc_auc_score(y_test, pred),    # how the notebook computes it (from 0/1 predictions)
    "roc_auc_proba": roc_auc_score(y_test, proba),    # from probabilities, the usual definition
}
print({k: round(v, 4) for k, v in metrics.items()})

# --- Export for the browser ---
vocab = vectorizer.get_feature_names_out()
in_vocab = set(vocab)
# The browser can't run WordNet, so ship a lookup of every word form seen in the corpus
# whose lemma differs from itself and lands on a vocabulary word.
lemma_map = {}
for w in surface_counts:
    lem = lemmatizer.lemmatize(w)
    if lem != w and lem in in_vocab:
        lemma_map[w] = lem

out = {
    "about": "Logistic regression on TF-IDF (max_features=5000) from NLP_Algo_Harness.ipynb, trained on IMDB 50k reviews.",
    "vocab": list(vocab),
    "idf": [round(float(v), 6) for v in vectorizer.idf_],
    "coef": [round(float(v), 6) for v in model.coef_[0]],
    "intercept": float(model.intercept_[0]),
    "stopwords": sorted(stop_words),
    "lemmas": lemma_map,
    "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    "n_train": int(X_train_tfidf.shape[0]),
    "n_test": int(X_test_tfidf.shape[0]),
}
Path("data").mkdir(exist_ok=True)
with open("data/model.json", "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))

# A few test cases the website must reproduce exactly
checks = ["The acting was wonderful but the ending dragged on forever.",
          "Worst movie I have seen in years, a total waste of money!",
          "Visually stunning <br /> and the soundtrack gave me chills. 10/10 https://t.co/x"]
with open("data/model_checks.json", "w", encoding="utf-8") as f:
    json.dump([{"text": t, "processed": preprocess(t), "p_positive": float(model.predict_proba(vectorizer.transform([preprocess(t)]))[0, 1])} for t in checks], f, indent=1)
print("Wrote data/model.json and data/model_checks.json")
