import io
import json
import os
import re
import time

import pytest

from app import create_app
from conftest import ROOT, good_row, make_config, post_csv


# ---------------------------------------------------------------- startup
def test_app_starts_with_valid_models(client):
    r = client.get('/api/health').get_json()
    assert r['ok'] and r['data']['status'] == 'ok'
    assert r['data']['url_model']['loaded'] and r['data']['file_model']['loaded']


def test_app_survives_missing_models_and_reports_it(tmp_path):
    app = create_app(make_config(str(tmp_path / 'nope'), tmp_path))
    c = app.test_client()
    h = c.get('/api/health').get_json()['data']
    assert h['status'] == 'degraded' and 'url_model.joblib' in h['url_model']['error'] and 'malware_model_card.json' in h['file_model']['error']
    r = c.post('/api/predict-url', json={'url': 'example.com'})
    assert r.status_code == 503 and r.get_json()['error']['code'] == 'model_unavailable'
    assert post_csv(c, b'a,b\n1,2\n').status_code == 503
    assert c.get('/').status_code == 200  # the page still loads and shows the banner


@pytest.mark.integration
def test_app_starts_with_real_models(real_app):
    data = real_app.test_client().get('/api/health').get_json()['data']
    assert data['status'] == 'ok', data


# ---------------------------------------------------------------- URL endpoint
def test_url_ok(client):
    r = client.post('/api/predict-url', json={'url': 'https://paypal-login-verify.example-bad.xyz/update'})
    d = r.get_json()
    assert r.status_code == 200 and d['ok']
    assert d['data']['prediction'] in {'Safe URL', 'Malicious URL'} and 0 <= d['data']['malicious_score'] <= 1


@pytest.mark.parametrize('payload,code', [({'url': ''}, 'invalid_url'), ({'url': '   '}, 'invalid_url'), ({'url': 'not a url'}, 'invalid_url'),
                                          ({'url': 'javascript:alert(1)'}, 'invalid_url'), ({'url': 5}, 'invalid_url'),
                                          ({'other': 'x'}, 'bad_request'), ([], 'bad_request')])
def test_url_bad_input(client, payload, code):
    r = client.post('/api/predict-url', json=payload)
    assert r.status_code == 400 and r.get_json()['error']['code'] == code


def test_url_requires_json(client):
    r = client.post('/api/predict-url', data='url=example.com', content_type='application/x-www-form-urlencoded')
    assert r.status_code == 415
    assert client.post('/api/predict-url', data='{broken', content_type='application/json').status_code == 400


def test_url_is_never_fetched(client, monkeypatch):
    import socket
    def boom(*a, **k): raise AssertionError('network access attempted')
    monkeypatch.setattr(socket, 'getaddrinfo', boom)
    monkeypatch.setattr(socket.socket, 'connect', boom)
    assert client.post('/api/predict-url', json={'url': 'http://example.invalid/x'}).status_code == 200


# ---------------------------------------------------------------- CSV endpoint
def test_csv_ok_and_download(client, csv_bytes):
    r = post_csv(client, csv_bytes(), 'my scan.csv')
    d = r.get_json()['data']
    assert r.status_code == 200 and d['summary']['rows'] == 2 and d['download']['filename'] == 'my_scan_predictions.csv'
    dl = client.get(d['download']['url'] + '?name=' + d['download']['filename'])
    assert dl.status_code == 200 and dl.mimetype == 'text/csv'
    assert 'attachment' in dl.headers['Content-Disposition'] and 'my_scan_predictions.csv' in dl.headers['Content-Disposition']
    lines = dl.data.decode().strip().splitlines()
    assert lines[0].endswith('malware_score,prediction') and len(lines) == 3


def test_csv_with_bom_and_crlf(client, csv_bytes):
    raw = b'\xef\xbb\xbf' + csv_bytes().replace(b'\n', b'\r\n')
    assert post_csv(client, raw).status_code == 200


def test_missing_file_field(client):
    r = client.post('/api/predict-file', data={}, content_type='multipart/form-data')
    assert r.status_code == 400 and r.get_json()['error']['code'] == 'no_file'


def test_unsupported_extension(client, csv_bytes):
    for name in ('scan.txt', 'scan.exe', 'scan.csv.exe', 'scan'):
        r = post_csv(client, csv_bytes(), name)
        assert r.status_code == 415 and r.get_json()['error']['code'] == 'unsupported_file_type'


def test_empty_file(client):
    assert post_csv(client, b'').get_json()['error']['code'] == 'empty_file'
    assert post_csv(client, b'  \n ').get_json()['error']['code'] == 'empty_file'


def test_binary_file_renamed_to_csv_is_rejected(client):
    r = post_csv(client, b'MZ\x90\x00\x03\x00\x00\x00' + os.urandom(300))
    assert r.status_code == 400 and r.get_json()['error']['code'] == 'not_text'


def test_non_utf8_file(client):
    r = post_csv(client, 'a,b\n\xe9,2\n'.encode('latin-1'))
    assert r.status_code == 400 and r.get_json()['error']['code'] == 'bad_encoding'


def test_malformed_csv(client):
    r = post_csv(client, b'a,b,c\n1,2,3,4,5\n"unterminated,1\n')
    assert r.status_code in (400, 422)
    assert r.get_json()['ok'] is False


def test_header_only_csv(client, csv_bytes):
    r = post_csv(client, csv_bytes(rows=[]) if False else (','.join(good_row().keys()) + '\n').encode())
    assert r.status_code == 422 and 'no data rows' in r.get_json()['error']['message']


def test_wrong_column_counts(client, csv_bytes):
    few = {k: v for k, v in list(good_row().items())[:20]}
    r = post_csv(client, csv_bytes(rows=[few]))
    assert r.status_code == 422 and 'Missing 7 required column' in r.get_json()['error']['message']
    r = post_csv(client, csv_bytes(rows=[dict(good_row(), extra=1)]))
    assert r.status_code == 422 and 'Unexpected column' in r.get_json()['error']['message']


def test_non_numeric_and_missing_cells(client, csv_bytes):
    r = post_csv(client, csv_bytes(rows=[good_row(state='oops')]))
    assert r.status_code == 422 and 'non-numeric' in r.get_json()['error']['message']
    r = post_csv(client, csv_bytes(rows=[good_row(state=None)]))
    assert r.status_code == 422 and 'empty cell' in r.get_json()['error']['message']


def test_too_many_rows(client, csv_bytes):
    r = post_csv(client, csv_bytes(rows=[good_row()] * 1001))
    assert r.status_code == 422 and 'limit is 1,000' in r.get_json()['error']['message']


def test_upload_size_limit(toy_model_dir, tmp_path, csv_bytes):
    c = create_app(make_config(toy_model_dir, tmp_path, MAX_CONTENT_LENGTH=500, MAX_UPLOAD_MB=0)).test_client()
    r = post_csv(c, csv_bytes(rows=[good_row()] * 20))
    assert r.status_code == 413 and r.get_json()['error']['code'] == 'file_too_large'


def test_uploaded_file_is_not_stored_and_results_expire(toy_model_dir, tmp_path, csv_bytes):
    app = create_app(make_config(toy_model_dir, tmp_path, RESULT_TTL_SECONDS=1))
    c = app.test_client()
    first = post_csv(c, csv_bytes()).get_json()['data']['download']['url']
    stored = [p for p in os.listdir(tmp_path)]
    assert len(stored) == 1 and stored[0].endswith('.csv') and 'scan' not in stored[0]  # only the generated result, random name
    time.sleep(1.2)
    assert c.get(first).status_code == 404
    post_csv(c, csv_bytes())  # triggers cleanup of the expired file
    assert len(os.listdir(tmp_path)) == 1


@pytest.mark.parametrize('token', ['../../etc/passwd', 'x' * 32, 'A' * 32, '0' * 31, '0' * 32])
def test_download_rejects_bad_tokens(client, token):
    assert client.get(f'/api/download/{token}').status_code == 404


# ---------------------------------------------------------------- misc / integration
def test_unknown_api_route_returns_json_404_and_wrong_method_405(client):
    r = client.get('/api/nothing')
    assert r.status_code == 404 and r.get_json()['ok'] is False
    assert client.get('/api/predict-url').status_code == 405


def test_security_headers(client):
    h = client.get('/').headers
    assert h['X-Content-Type-Options'] == 'nosniff' and "default-src 'self'" in h['Content-Security-Policy']


def test_model_info(client):
    d = client.get('/api/model-info').get_json()['data']
    assert d['url']['kind'] == 'url' and d['file']['kind'] == 'malware'


def test_frontend_matches_backend(app, client):
    js = open(os.path.join(ROOT, 'static', 'js', 'app.js'), encoding='utf-8').read()
    html = client.get('/').get_data(as_text=True)
    routes = {r.rule for r in app.url_map.iter_rules()}
    called = set(re.findall(r"api\('(/api/[a-z\-]+)", js))
    assert called == {'/api/predict-url', '/api/predict-file', '/api/health', '/api/model-info'}
    assert called <= routes
    ids = set(re.findall(r"\$\('([a-z\-]+)'\)", js))
    missing = [i for i in ids if f'id="{i}"' not in html]
    assert not missing, f'ids used by app.js but absent from index.html: {missing}'
    assert '/static/js/app.js' in html and '/static/css/style.css' in html
    assert client.get('/static/js/app.js').status_code == 200 and client.get('/static/css/style.css').status_code == 200
    assert '<script>' not in html  # no inline scripts (CSP)


def test_frontend_never_writes_user_data_with_innerhtml():
    js = open(os.path.join(ROOT, 'static', 'js', 'app.js'), encoding='utf-8').read()
    assert 'innerHTML' not in js and 'insertAdjacentHTML' not in js


@pytest.mark.integration
def test_end_to_end_with_real_models(real_app):
    c = real_app.test_client()
    r = c.post('/api/predict-url', json={'url': 'https://www.wikipedia.org/wiki/Python'})
    assert r.status_code == 200 and r.get_json()['data']['label'] == 'safe'
    raw = open(os.path.join(ROOT, 'sample_data', 'TestBinary.csv'), 'rb').read()
    r = post_csv(c, raw, 'TestBinary.csv')
    assert r.status_code == 200 and r.get_json()['data']['summary']['rows'] == 11
