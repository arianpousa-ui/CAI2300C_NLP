# Crowd Read

**Read the crowd before the box office does.** A landing page and live demo for an NLP class project (CAI2300C Natural Language Processing).

A person writing a review already knows whether they liked the film, so classifying one review tells them nothing new. The people who need sentiment analysis are the ones facing thousands of unlabeled comments: studio marketing teams, theater programmers and streaming services reading social posts, forum threads and survey answers that come with no star rating. Crowd Read scores all of them at once and reports:

- **Audience score:** the share of comments that read as positive
- **What people are talking about:** acting, story, visuals, pacing, music, ending, dialogue and characters, each scored clause by clause
- **Answer these first:** the most clearly negative comments
- **Read these yourself:** the comments the model can't decide on
- **Why the model said that:** the push from every word in a comment

## The model

It comes from [`notebooks/NLP_Algo_Harness.ipynb`](notebooks/NLP_Algo_Harness.ipynb): IMDB 50k movie reviews → clean text → tokenize → remove stop words → WordNet lemmatize → TF-IDF (5,000 words) → eight classifiers compared on the same 10,000 test reviews. Logistic regression won (88.5% accuracy, 0.887 F1).

[`train/export_model.py`](train/export_model.py) reruns that pipeline and writes `data/model.json` (vocabulary, IDF values, weights and a lemma lookup). [`assets/model.js`](assets/model.js) applies the same steps in the browser. On 1,500 held-out reviews, the browser and scikit-learn give the same label on all but one, with an average probability difference of 0.002.

To retrain (Windows):

```
pip install pandas scikit-learn nltk
python train\export_model.py "path\to\IMDB Dataset.csv"
```

## Run locally

The page loads its model with `fetch`, so serve the folder instead of opening the file directly:

```
python -m http.server 8000
```

Then open http://localhost:8000.

## Deploy

Static site, no build step. On Netlify, import this repository; `netlify.toml` publishes the repository root.

## Files

| Path | What it is |
| --- | --- |
| `index.html`, `assets/style.css`, `assets/app.js` | The landing page and pulse-board demo |
| `assets/model.js` | Browser version of the notebook's preprocessing and prediction |
| `data/model.json` | Exported logistic regression model (~200 KB) |
| `data/sample_comments.json` | 50 invented comments about a fictional film, *The Lantern Keeper* |
| `data/model_checks.json` | Reference predictions from Python, used to check the browser model |
| `train/export_model.py` | Retrains the model and writes `data/model.json` |
| `notebooks/NLP_Algo_Harness.ipynb` | The course notebook the model comes from |
