import os

import joblib
import pytest

from detectors.errors import InvalidURLError, ModelLoadError
from detectors.url_detector import URLDetector, validate_url
from detectors.url_features import normalize_url, host_of, registered_domain
from conftest import ROOT


@pytest.mark.parametrize('raw,expected', [
    ('https://www.Google.com', 'Google.com/'),
    ('http://user:pw@example.com/a?b=1#frag', 'example.com/a?b=1'),
    ('example.com/page', 'example.com/page'),
    ('  WWW.example.com  ', 'example.com/'),
    ('ftp://files.example.org/x.zip', 'files.example.org/x.zip'),
])
def test_normalize_url(raw, expected):
    assert normalize_url(raw) == expected


def test_host_and_registered_domain():
    assert host_of('Sub.Example.co.uk:8080/p') == 'sub.example.co.uk'
    assert registered_domain('a.b.example.co.uk') == 'example.co.uk'
    assert registered_domain('mail.example.com') == 'example.com'


@pytest.mark.parametrize('raw', ['example.com', 'https://example.com/path?q=1', 'http://192.168.1.10/login.php', 'sub.domain.co.uk/a',
                                 'ftp://files.example.org/x'])
def test_validate_accepts_reasonable_urls(raw):
    assert validate_url(raw) == raw.strip()


@pytest.mark.parametrize('raw', ['', '   ', None, 123, 'not a url', 'localhost', 'http://', 'javascript:alert(1)', 'data://text/plain,hi',
                                 'file:///etc/passwd', 'http://999.1.1.1/', 'http://[::1]/', 'exa mple.com', 'example.com/\x00', 'a' * 3000])
def test_validate_rejects_bad_input(raw):
    with pytest.raises(InvalidURLError):
        validate_url(raw)


def test_predict_result_shape_and_score_matches_bad_class(toy_model_dir):
    det = URLDetector(toy_model_dir)
    res = det.predict('https://paypal-login-verify.example-bad.xyz/update')
    expected = det.pipeline.predict_proba([normalize_url('https://paypal-login-verify.example-bad.xyz/update')])[0][list(det.clf.classes_).index('bad')]
    assert res['malicious_score'] == pytest.approx(expected, abs=1e-4)
    assert res['label'] == 'malicious' and res['prediction'] == 'Malicious URL'
    assert set(res) >= {'prediction', 'label', 'malicious_score', 'threshold', 'risk_band', 'normalized_url', 'signals'}
    assert det.predict('wikipedia.org/wiki/Python')['label'] == 'safe'


def test_threshold_controls_verdict(toy_model_dir):
    assert URLDetector(toy_model_dir, threshold=0.0).predict('wikipedia.org/wiki/Python')['label'] == 'malicious'
    assert URLDetector(toy_model_dir, threshold=1.01).predict('paypal-login-verify.example-bad.xyz/update')['label'] == 'safe'


def test_missing_model_files_raise_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match='url_model.joblib'):
        URLDetector(str(tmp_path))


def test_corrupt_model_file_raises(toy_model_dir, tmp_path):
    (tmp_path / 'url_model.joblib').write_bytes(b'not a pickle')
    (tmp_path / 'url_model_card.json').write_text('{}')
    with pytest.raises(ModelLoadError, match='Could not load'):
        URLDetector(str(tmp_path))


def test_model_without_bad_class_is_rejected(toy_model_dir, tmp_path):
    pipe = joblib.load(os.path.join(toy_model_dir, 'url_model.joblib'))
    pipe.named_steps['clf'].classes_ = pipe.named_steps['clf'].classes_.astype(object)
    pipe.named_steps['clf'].classes_[:] = ['x', 'y']
    joblib.dump(pipe, tmp_path / 'url_model.joblib')
    (tmp_path / 'url_model_card.json').write_text('{}')
    with pytest.raises(ModelLoadError, match='expected structure'):
        URLDetector(str(tmp_path))


@pytest.mark.integration
class TestRealURLModel:
    """Smoke tests on the shipped model. They check wiring and obvious cases, they do not measure accuracy."""

    @pytest.fixture(scope='class')
    def det(self):
        return URLDetector(os.path.join(ROOT, 'model'))

    def test_scheme_and_www_do_not_change_the_verdict(self, det):
        a = det.predict('https://www.wikipedia.org/wiki/Python')['malicious_score']
        b = det.predict('wikipedia.org/wiki/Python')['malicious_score']
        assert a == b

    def test_obvious_cases(self, det):
        assert det.predict('https://en.wikipedia.org/wiki/Python')['label'] == 'safe'
        assert det.predict('http://paypal-secure-login.verify-account.xyz/update')['label'] == 'malicious'

    def test_sample_file_sanity(self, det):
        """Sanity check on sample_data/TestURL.csv (rows come from the training file, so this is wiring, not accuracy).

        Most 'bad' rows in that file are bare domains, which are the hardest case once URLs are normalised, so only
        the 'good' rows and the 'bad' rows that include a path are asserted.
        """
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, 'sample_data', 'TestURL.csv'))
        flagged = pd.Series([det.predict(u)['label'] == 'malicious' for u in df.url], index=df.index)
        good = df.label == 'good'
        assert (~flagged[good]).mean() >= 0.9
        bad_with_path = (df.label == 'bad') & df.url.str.contains('/') & ~df.url.str.endswith('/')
        assert flagged[bad_with_path].mean() >= 0.4
