"""Domain-grouped evaluation of the URL model: all URLs of one domain stay on the same side of the split."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from detectors.url_features import normalize_url, host_of, registered_domain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

d = pd.read_csv(sys.argv[1]).dropna().drop_duplicates()
d = d.drop_duplicates('url', keep=False)  # drop URLs that carry conflicting labels
d['norm'] = d.url.map(normalize_url)
d['dom'] = d.norm.map(lambda s: registered_domain(host_of(s)))
y = (d.label == 'bad').astype(int).values
tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=42).split(d, y, d.dom))
print('rows', len(d), 'domains', d.dom.nunique(), 'train', len(tr), 'test', len(te), 'overlap domains', len(set(d.dom.iloc[tr]) & set(d.dom.iloc[te])))

def run(name, text, **lr_kw):
    v = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=5, sublinear_tf=True)
    A = v.fit_transform(text.iloc[tr]); B = v.transform(text.iloc[te])
    m = LogisticRegression(max_iter=1000, **lr_kw).fit(A, y[tr]); p = m.predict_proba(B)[:, 1]
    print(f'\n== {name}  ROC-AUC {roc_auc_score(y[te], p):.4f}')
    for t in (.5,):
        print(confusion_matrix(y[te], p >= t)); print(classification_report(y[te], p >= t, target_names=['good','bad'], digits=4))
    return roc_auc_score(y[te], p)
run('raw URL text, LR (original recipe, grouped split)', d.url)
run('normalised URL text, LR', d.norm)
run('normalised URL text, LR class_weight=balanced', d.norm, class_weight='balanced')
