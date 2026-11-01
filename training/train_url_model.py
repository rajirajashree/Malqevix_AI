"""Train the URL model: normalised URL -> char 3-5 gram TF-IDF -> Logistic Regression.

Usage:  python training/train_url_model.py path/to/data_url.csv
Writes: model/url_model.joblib  (sklearn Pipeline, vectorizer + classifier fitted together)
        model/url_model_card.json (label order, metrics, library versions)

Metrics are measured on a held-out set of DOMAINS the model never saw (no domain appears on both sides).
The shipped model is then refitted on all de-duplicated rows with the same recipe.
"""
import sys, os, json, datetime
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from detectors.url_features import normalize_url, host_of, registered_domain

def make_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=5, sublinear_tf=True, dtype=np.float32)),
        ('clf', LogisticRegression(max_iter=1000)),
    ])

def expected_calibration_error(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def main(path):
    raw = pd.read_csv(path)
    n_raw = len(raw)
    d = raw.dropna().drop_duplicates().drop_duplicates('url', keep=False)  # exact duplicates + conflicting labels removed
    assert set(d.label.unique()) == {'good', 'bad'}, d.label.unique()
    d = d.assign(norm=d.url.map(normalize_url))
    d['dom'] = d.norm.map(lambda s: registered_domain(host_of(s)))
    y = d.label.values
    yb = (y == 'bad').astype(int)

    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(d, yb, d.dom))
    assert not set(d.dom.iloc[tr]) & set(d.dom.iloc[te])
    hold = make_pipeline().fit(d.norm.iloc[tr], y[tr])
    bad_i = list(hold.classes_).index('bad')
    p = hold.predict_proba(d.norm.iloc[te])[:, bad_i]; yt = yb[te]; pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    metrics = dict(
        split='GroupShuffleSplit by registered domain, 80/20, seed 42', test_rows=int(len(te)), test_domains=int(d.dom.iloc[te].nunique()),
        threshold=0.5, accuracy=float((pred == yt).mean()), precision_bad=float(precision_score(yt, pred)),
        recall_bad=float(recall_score(yt, pred)), f1_bad=float(f1_score(yt, pred)), roc_auc=float(roc_auc_score(yt, p)),
        false_positive_rate=float(fp / (fp + tn)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        brier=float(brier_score_loss(yt, p)), expected_calibration_error=expected_calibration_error(yt, p))
    print(json.dumps(metrics, indent=2))

    final = make_pipeline().fit(d.norm, y)
    os.makedirs(os.path.join(ROOT, 'model'), exist_ok=True)
    joblib.dump(final, os.path.join(ROOT, 'model', 'url_model.joblib'), compress=3)
    card = dict(kind='url', created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                classes=[str(c) for c in final.classes_], malicious_class='bad', default_threshold=0.5,
                preprocessing='detectors/url_features.normalize_url (strip scheme, credentials, www., fragment; ensure "/" after host)',
                training_rows_raw=int(n_raw), training_rows_used=int(len(d)), training_domains=int(d.dom.nunique()),
                class_counts={k: int(v) for k, v in pd.Series(y).value_counts().items()},
                sklearn_version=sklearn.__version__, numpy_version=np.__version__, held_out_metrics=metrics)
    json.dump(card, open(os.path.join(ROOT, 'model', 'url_model_card.json'), 'w'), indent=2)
    print('saved model/url_model.joblib')

if __name__ == '__main__':
    main(sys.argv[1])
