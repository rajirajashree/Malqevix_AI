"""Malicious-URL detector. Works on the URL *string* only: nothing is ever fetched or opened."""
import json
import logging
import os
import re

import joblib
import numpy as np

from .errors import InvalidURLError, ModelLoadError
from .url_features import normalize_url, host_of

log = logging.getLogger(__name__)

_SCHEME = re.compile(r'^([a-zA-Z][a-zA-Z0-9+.\-]*)://')
_ALLOWED_SCHEMES = {'http', 'https', 'ftp'}
_IPV4 = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')
_HOST = re.compile(r'^(?=.{1,253}$)(?:[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?\.)+[a-z0-9-]{2,63}$')


def validate_url(raw, max_length=2048):
    """Return the stripped URL or raise InvalidURLError with a message that tells the user what to fix."""
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidURLError('Enter a URL to analyse.')
    s = raw.strip()
    if len(s) > max_length:
        raise InvalidURLError(f'The URL is longer than {max_length} characters.')
    if re.search(r'[\s\x00-\x1f\x7f]', s):
        raise InvalidURLError('The URL contains spaces or control characters. Remove them and try again.')
    m = _SCHEME.match(s)
    if m and m.group(1).lower() not in _ALLOWED_SCHEMES:
        raise InvalidURLError(f'Only http, https and ftp URLs can be analysed (got "{m.group(1).lower()}").')
    host = host_of(normalize_url(s))
    if host.startswith('['):
        raise InvalidURLError('IPv6 addresses are not supported. Use a domain name or an IPv4 address.')
    try:
        host_ascii = host.encode('idna').decode('ascii')
    except UnicodeError:
        raise InvalidURLError('The domain name is not valid.')
    if _IPV4.match(host_ascii):
        if any(int(p) > 255 for p in host_ascii.split('.')):
            raise InvalidURLError('The IP address is not valid.')
    elif not _HOST.match(host_ascii):
        raise InvalidURLError('That does not look like a web address. Use a form like example.com/page or https://example.com/page.')
    return s


class URLDetector:
    def __init__(self, model_dir, threshold=0.5):
        self.threshold = float(threshold)
        model_path = os.path.join(model_dir, 'url_model.joblib')
        card_path = os.path.join(model_dir, 'url_model_card.json')
        for p in (model_path, card_path):
            if not os.path.isfile(p):
                raise ModelLoadError(f'Missing URL model file: {os.path.basename(p)} (expected in {model_dir}). '
                                     'Run training/train_url_model.py or copy the file into the model folder.')
        try:
            self.pipeline = joblib.load(model_path)
            with open(card_path, encoding='utf-8') as fh:
                self.card = json.load(fh)
        except Exception as exc:  # unreadable pickle, version mismatch, bad JSON ...
            raise ModelLoadError(f'Could not load the URL model: {exc}. If this mentions scikit-learn or numpy versions, '
                                 'install the versions in requirements.txt or retrain with training/train_url_model.py.') from exc
        try:
            self.tfidf = self.pipeline.named_steps['tfidf']
            self.clf = self.pipeline.named_steps['clf']
            self.bad_index = list(self.clf.classes_).index('bad')
        except (KeyError, ValueError, AttributeError) as exc:
            raise ModelLoadError('The URL model file does not have the expected structure (tfidf + clf with a "bad" class).') from exc
        vocab = self.tfidf.vocabulary_
        self._terms = np.empty(len(vocab), dtype=object)
        for term, i in vocab.items():
            self._terms[i] = term
        import sklearn
        if self.card.get('sklearn_version') and self.card['sklearn_version'] != sklearn.__version__:
            log.warning('URL model was trained with scikit-learn %s but %s is installed', self.card['sklearn_version'], sklearn.__version__)

    def predict(self, raw_url, max_length=2048):
        url = validate_url(raw_url, max_length)
        normalized = normalize_url(url)
        score = float(self.pipeline.predict_proba([normalized])[0][self.bad_index])
        malicious = score >= self.threshold
        if score >= self.threshold:
            band = 'high'
        elif score >= 0.2:
            band = 'elevated'
        else:
            band = 'low'
        return {
            'prediction': 'Malicious URL' if malicious else 'Safe URL',
            'label': 'malicious' if malicious else 'safe',
            'malicious_score': round(score, 4),
            'threshold': self.threshold,
            'risk_band': band,
            'normalized_url': normalized,
            'bare_domain': '/' not in url.split('://', 1)[-1].rstrip('/') and '?' not in url,
            'signals': self._signals(normalized),
        }

    def _signals(self, normalized, top=5):
        """Character patterns in this URL that push the score toward 'malicious' the most (informational)."""
        x = self.tfidf.transform([normalized])
        sign = 1.0 if self.bad_index == 1 else -1.0
        contrib = x.multiply(self.clf.coef_[0] * sign).tocoo()
        pairs = sorted(zip(contrib.col, contrib.data), key=lambda t: -t[1])
        return [{'pattern': str(self._terms[c]), 'weight': round(float(v), 3)} for c, v in pairs[:top] if v > 0]

    @property
    def info(self):
        c = self.card
        return {'kind': 'url', 'algorithm': 'TF-IDF (character 3-5 grams) + Logistic Regression', 'created': c.get('created'),
                'threshold': self.threshold, 'held_out_metrics': c.get('held_out_metrics'),
                'training_rows_used': c.get('training_rows_used')}
