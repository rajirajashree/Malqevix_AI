"""Honest evaluation of the CSV malware model.

The dataset has 100 apps (hash) x 1000 consecutive 1 ms snapshots each. The original notebook split
ROWS at random, so snapshots of the same app land in both train and test (leakage). Here every app is
kept entirely on one side of the split (StratifiedGroupKFold on `hash`).
"""
import os, sys, json, numpy as np, pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf

DATA = sys.argv[1]
FEATURES = ['millisecond','state','usage_counter','prio','static_prio','normal_prio','policy','vm_pgoff','task_size',
 'cached_hole_size','free_area_cache','mm_users','map_count','hiwater_rss','total_vm','reserved_vm','nr_ptes','end_data',
 'last_interval','nivcsw','min_flt','fs_excl_counter','lock','stime','gtime','cgtime','signal_nvcsw']
df = pd.read_csv(DATA)
X = df[FEATURES].values.astype('float64'); y = (df.classification == 'malware').astype(int).values; g = df.hash.values

def mlp():
    tf.keras.utils.set_random_seed(1)
    m = tf.keras.Sequential([tf.keras.Input(shape=(27,))] + [tf.keras.layers.Dense(50, activation='relu') for _ in range(6)] + [tf.keras.layers.Dense(2, activation='softmax')])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def fit_predict(name, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); a, b = sc.transform(Xtr), sc.transform(Xte)
    if name == 'keras_mlp (original architecture)':
        m = mlp(); m.fit(a, ytr, batch_size=100, epochs=20, verbose=0); return m.predict(b, verbose=0)[:, 1]
    est = {'logistic_regression': LogisticRegression(max_iter=2000),
           'random_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=1),
           'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=1)}[name]
    return est.fit(a, ytr).predict_proba(b)[:, 1]

names = ['logistic_regression','random_forest','hist_gradient_boosting','keras_mlp (original architecture)']
oof = {n: np.zeros(len(y)) for n in names}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
for k, (tr, te) in enumerate(cv.split(X, y, g)):
    assert not set(g[tr]) & set(g[te])
    for n in names: oof[n][te] = fit_predict(n, X[tr], y[tr], X[te])
    print('fold', k, 'done', flush=True)

rows = {}
for n in names:
    p = oof[n]; yh = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    # app-level: average probability over each app's 1000 rows
    ag = pd.DataFrame({'g': g, 'p': p, 'y': y}).groupby('g').agg(p=('p','mean'), y=('y','first'))
    rows[n] = dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh), recall=recall_score(y, yh), f1=f1_score(y, yh),
                   roc_auc=roc_auc_score(y, p), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                   app_level_accuracy=accuracy_score(ag.y, (ag.p >= .5).astype(int)))
out = pd.DataFrame(rows).T.round(4); print(out.to_string())
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grouped_cv_results.json'), 'w'), indent=2)
