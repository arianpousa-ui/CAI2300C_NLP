/* Browser port of NLP_Algo_Harness.ipynb's prediction path.
   Mirrors: clean_text -> word_tokenize -> stop words -> WordNet lemmatize -> TF-IDF (5,000 words) -> Logistic Regression. */
(function (root) {
  const PUNCT = /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g;   // Python's string.punctuation
  const TOKEN = /[\p{L}\p{N}_]+/gu;                         // word_tokenize on punctuation-free text
  const SKLEARN_TOKEN = /\b\w\w+\b/gu;                      // TfidfVectorizer's default token_pattern

  function cleanText(text) {
    let t = String(text).toLowerCase();
    t = t.replace(/<.*?>/g, '');
    t = t.replace(/https?:\/\/\S+|www\.\S+/g, '');
    t = t.replace(PUNCT, '');
    t = t.replace(/\d+/g, '');
    return t;
  }

  function makeModel(json) {
    const vocab = new Map(json.vocab.map((w, i) => [w, i]));
    const stop = new Set(json.stopwords);
    const lemmas = json.lemmas;
    const idf = Float64Array.from(json.idf);
    const coef = Float64Array.from(json.coef);
    const b = json.intercept;

    function lemma(w) {
      if (lemmas[w]) return lemmas[w];
      if (!vocab.has(w) && w.length > 3 && w.endsWith('s') && vocab.has(w.slice(0, -1))) return w.slice(0, -1);
      return w;
    }

    function preprocess(text) {
      const tokens = (cleanText(text).match(TOKEN) || []).filter(w => !stop.has(w));
      return tokens.map(lemma).join(' ');
    }

    function vectorize(processed) {
      const counts = new Map();
      for (const t of processed.match(SKLEARN_TOKEN) || []) {
        const j = vocab.get(t);
        if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
      }
      let norm = 0;
      const feats = [];
      for (const [j, c] of counts) { const v = c * idf[j]; feats.push([j, v]); norm += v * v; }
      norm = Math.sqrt(norm) || 1;
      return feats.map(([j, v]) => [j, v / norm]);
    }

    function score(text) {
      const processed = preprocess(text);
      const x = vectorize(processed);
      let z = b;
      const contrib = x.map(([j, v]) => { const c = v * coef[j]; z += c; return { word: json.vocab[j], value: c }; });
      const p = 1 / (1 + Math.exp(-z));
      return { p, processed, contrib, known: x.length };
    }

    return { preprocess, score, metrics: json.metrics, nTrain: json.n_train, nTest: json.n_test, vocabSize: json.vocab.length };
  }

  root.ReelModel = { makeModel, cleanText };
})(typeof window !== 'undefined' ? window : globalThis);
if (typeof module !== 'undefined') module.exports = globalThis.ReelModel;
