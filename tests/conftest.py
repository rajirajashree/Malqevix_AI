import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from app import create_app  # noqa: E402
from config import TestConfig  # noqa: E402
from detectors.file_features import FEATURE_COLUMNS  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, 'sample_data')


def good_row(**over):
    row = {c: 0 for c in FEATURE_COLUMNS}
    row.update(over)
    return row


@pytest.fixture(scope='session')
def toy_model_dir(tmp_path_factory):
    """Tiny, fast models with the same file layout/contract as the real ones (used by unit tests)."""
    d = tmp_path_factory.mktemp('toy_models')
    urls = ['paypal-login-verify.example-bad.xyz/update', 'secure-bank-verify.evil.ru/login.php', 'free-crack-download.biz/setup.exe',
            'wikipedia.org/wiki/Python', 'github.com/user/repo', 'news.example.com/2020/story.html'] * 5
    labels = ['bad'] * 3 + ['good'] * 3
    pipe = Pipeline([('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5))), ('clf', LogisticRegression(max_iter=500))])
    pipe.fit(urls, labels * 5)
    joblib.dump(pipe, d / 'url_model.joblib')
    (d / 'url_model_card.json').write_text(json.dumps({'kind': 'url', 'held_out_metrics': {'accuracy': 1.0}}))

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, 27)), columns=FEATURE_COLUMNS)
    y = (X['state'] > 0.5).astype(int)  # class 1 (= Malware) whenever `state` is high
    joblib.dump(RandomForestClassifier(20, random_state=0).fit(X, y), d / 'malware_rf.joblib')
    (d / 'malware_model_card.json').write_text(json.dumps({'kind': 'malware', 'features': FEATURE_COLUMNS, 'grouped_cv_metrics': {'accuracy': 1.0}}))
    return str(d)


def make_config(model_dir, upload_dir, **extra):
    attrs = {'MODEL_DIR': model_dir, 'UPLOAD_DIR': str(upload_dir), 'MAX_ROWS': 1000}
    attrs.update(extra)
    return type('Cfg', (TestConfig,), attrs)


@pytest.fixture()
def app(toy_model_dir, tmp_path):
    return create_app(make_config(toy_model_dir, tmp_path))


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope='session')
def real_app(tmp_path_factory):
    return create_app(make_config(os.path.join(ROOT, 'model'), tmp_path_factory.mktemp('uploads'), MAX_ROWS=200000))


@pytest.fixture()
def csv_bytes():
    def build(rows=None, **kw):
        rows = rows if rows is not None else [good_row(state=1000), good_row(state=0)]
        return pd.DataFrame(rows).to_csv(index=False, **kw).encode()
    return build


def post_csv(client, data, filename='scan.csv'):
    import io
    return client.post('/api/predict-file', data={'file': (io.BytesIO(data), filename)}, content_type='multipart/form-data')
