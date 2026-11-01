"""AI Malware Detection System - Flask application (application factory).

Run locally:   python app.py            (debug only if FLASK_DEBUG=1)
Windows prod:  waitress-serve --listen=0.0.0.0:8000 --call app:create_app
Linux prod:    gunicorn -c gunicorn.conf.py "app:create_app()"
"""
import io
import logging
import os
import re
import time
import uuid

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from pandas.errors import EmptyDataError, ParserError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import Config
from detectors.errors import FileValidationError, InvalidURLError, ModelLoadError
from detectors.file_detector import FileDetector
from detectors.url_detector import URLDetector

log = logging.getLogger('malware_detector')
_TOKEN = re.compile(r'^[a-f0-9]{32}$')


def ok(data, status=200):
    return jsonify({'ok': True, 'data': data}), status


def fail(code, message, status):
    return jsonify({'ok': False, 'error': {'code': code, 'message': message}}), status


def load_models(app):
    """Load every model once at startup. A failure is recorded (and reported by /api/health) instead of crashing the server."""
    app.extensions['models'] = {'url': None, 'file': None}
    app.extensions['model_errors'] = {'url': None, 'file': None}
    loaders = {
        'url': lambda: URLDetector(app.config['MODEL_DIR'], app.config['URL_THRESHOLD']),
        'file': lambda: FileDetector(app.config['MODEL_DIR'], app.config['MALWARE_ENGINE'], app.config['FILE_THRESHOLD']),
    }
    for name, loader in loaders.items():
        try:
            app.extensions['models'][name] = loader()
            log.info('%s model loaded', name)
        except ModelLoadError as exc:
            app.extensions['model_errors'][name] = str(exc)
            log.error('%s model NOT loaded: %s', name, exc)


def cleanup_results(app):
    """Delete generated result files older than RESULT_TTL_SECONDS."""
    folder, ttl, now = app.config['UPLOAD_DIR'], app.config['RESULT_TTL_SECONDS'], time.time()
    try:
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.endswith('.csv') and now - entry.stat().st_mtime > ttl:
                try:
                    os.remove(entry.path)
                except OSError:
                    log.warning('could not remove expired result file')
    except FileNotFoundError:
        pass


def create_app(config_object=None, load=True):
    app = Flask(__name__)
    app.config.from_object(config_object or Config)
    logging.basicConfig(level=app.config.get('LOG_LEVEL', 'INFO'), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    os.makedirs(app.config['UPLOAD_DIR'], exist_ok=True)
    if load:
        load_models(app)
    else:
        app.extensions['models'] = {'url': None, 'file': None}
        app.extensions['model_errors'] = {'url': 'not loaded', 'file': 'not loaded'}

    def models():
        return app.extensions['models']

    @app.after_request
    def security_headers(resp):
        resp.headers['X-Content-Type-Options'] = 'nosniff'
        resp.headers['X-Frame-Options'] = 'DENY'
        resp.headers['Referrer-Policy'] = 'no-referrer'
        resp.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if request.path.startswith('/api/'):
            resp.headers['Cache-Control'] = 'no-store'
        return resp

    # ------------------------------------------------------------------ pages
    @app.get('/')
    def home():
        return render_template('index.html', limits={
            'max_upload_mb': app.config['MAX_UPLOAD_MB'], 'max_rows': app.config['MAX_ROWS'], 'max_url_length': app.config['MAX_URL_LENGTH']})

    # ------------------------------------------------------------------ status
    @app.get('/api/health')
    def health():
        errs = app.extensions['model_errors']
        data = {'status': 'ok' if not any(errs.values()) else 'degraded',
                'url_model': {'loaded': models()['url'] is not None, 'error': errs['url']},
                'file_model': {'loaded': models()['file'] is not None, 'error': errs['file']}}
        return ok(data)

    @app.get('/api/model-info')
    def model_info():
        m = models()
        return ok({'url': m['url'].info if m['url'] else None, 'file': m['file'].info if m['file'] else None})

    # ------------------------------------------------------------------ URL
    @app.post('/api/predict-url')
    def predict_url():
        det = models()['url']
        if det is None:
            return fail('model_unavailable', app.extensions['model_errors']['url'] or 'The URL model is not loaded.', 503)
        if not request.is_json:
            return fail('unsupported_media_type', 'Send the request as JSON: {"url": "..."}.', 415)
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or 'url' not in body:
            return fail('bad_request', 'The JSON body must contain a "url" field.', 400)
        try:
            result = det.predict(body['url'], app.config['MAX_URL_LENGTH'])
        except InvalidURLError as exc:
            return fail('invalid_url', str(exc), 400)
        except Exception:
            log.exception('URL prediction failed')
            return fail('internal_error', 'The URL could not be analysed because of an internal error.', 500)
        log.info('url scored: label=%s score=%.3f length=%d', result['label'], result['malicious_score'], len(str(body['url'])))
        return ok(result)

    # ------------------------------------------------------------------ CSV
    @app.post('/api/predict-file')
    def predict_file():
        det = models()['file']
        if det is None:
            return fail('model_unavailable', app.extensions['model_errors']['file'] or 'The malware model is not loaded.', 503)
        upload = request.files.get('file')
        if upload is None or not upload.filename:
            return fail('no_file', 'Choose a CSV file to upload (form field "file").', 400)
        name = secure_filename(upload.filename) or 'upload.csv'
        if not name.lower().endswith('.csv'):
            return fail('unsupported_file_type', 'Only .csv files are accepted.', 415)
        raw = upload.read()
        if not raw.strip():
            return fail('empty_file', 'The file is empty.', 400)
        if b'\x00' in raw[:8192]:
            return fail('not_text', 'This does not look like a text CSV file (binary content detected). Files are never executed, but only CSV text is accepted.', 400)
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            return fail('bad_encoding', 'The file is not valid UTF-8 text. Re-save it as "CSV UTF-8".', 400)
        try:
            frame = pd.read_csv(io.StringIO(text))
        except EmptyDataError:
            return fail('empty_file', 'The file contains no data.', 400)
        except (ParserError, ValueError) as exc:
            return fail('malformed_csv', f'The CSV could not be parsed: {str(exc)[:200]}', 400)
        try:
            out, report = det.predict_frame(frame, app.config['MAX_ROWS'], app.config['PREVIEW_ROWS'])
        except FileValidationError as exc:
            return fail('invalid_csv', str(exc), 422)
        except Exception:
            log.exception('CSV prediction failed')
            return fail('internal_error', 'The file could not be scored because of an internal error.', 500)

        cleanup_results(app)
        token = uuid.uuid4().hex
        out.to_csv(os.path.join(app.config['UPLOAD_DIR'], token + '.csv'), index=False)
        stem = os.path.splitext(name)[0][:60] or 'results'
        report['download'] = {'url': f'/api/download/{token}', 'filename': f'{stem}_predictions.csv'}
        report['model'] = {'algorithm': det.algorithm, 'engine': det.engine}
        log.info('csv scored: rows=%d malware=%d', report['summary']['rows'], report['summary']['malware'])
        return ok(report)

    @app.get('/api/download/<token>')
    def download(token):
        if not _TOKEN.match(token):
            return fail('not_found', 'Result not found.', 404)
        path = os.path.join(app.config['UPLOAD_DIR'], token + '.csv')
        if not os.path.isfile(path) or time.time() - os.path.getmtime(path) > app.config['RESULT_TTL_SECONDS']:
            return fail('expired', 'This result has expired. Run the scan again.', 404)
        return send_file(path, mimetype='text/csv', as_attachment=True,
                         download_name=secure_filename(request.args.get('name', '')) or 'predictions.csv')

    # ------------------------------------------------------------------ errors
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return fail('file_too_large', f'The upload is larger than {app.config["MAX_UPLOAD_MB"]} MB.', 413)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if request.path.startswith('/api/'):
            return fail(exc.name.lower().replace(' ', '_'), exc.description, exc.code)
        return exc

    @app.errorhandler(Exception)
    def unhandled(exc):
        log.exception('unhandled error')
        if request.path.startswith('/api/'):
            return fail('internal_error', 'Unexpected server error.', 500)
        return 'Internal server error', 500

    return app


if __name__ == '__main__':
    application = create_app()
    application.run(host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', '5000')), debug=application.config['DEBUG'])
