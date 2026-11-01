"""Application configuration. Everything can be overridden with environment variables (or a .env file loaded by
your shell / hosting platform). Debug mode is OFF unless FLASK_DEBUG=1 is set explicitly."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _bool(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in {'1', 'true', 'yes', 'on'}


class Config:
    DEBUG = _bool('FLASK_DEBUG', False)
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-only-change-me')  # no sessions are used; set a real value in production anyway

    MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(BASE_DIR, 'model'))
    UPLOAD_DIR = os.environ.get('UPLOAD_DIR', os.path.join(BASE_DIR, 'uploads'))  # holds generated result CSVs only

    # 'sklearn' = retrained Random Forest (model/malware_rf.joblib), 'keras' = your original deep-learning model
    MALWARE_ENGINE = os.environ.get('MALWARE_ENGINE', 'sklearn').strip().lower()

    URL_THRESHOLD = float(os.environ.get('URL_THRESHOLD', '0.5'))
    FILE_THRESHOLD = float(os.environ.get('FILE_THRESHOLD', '0.5'))

    MAX_UPLOAD_MB = int(os.environ.get('MAX_UPLOAD_MB', '20'))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    MAX_ROWS = int(os.environ.get('MAX_ROWS', '200000'))
    MAX_URL_LENGTH = 2048
    RESULT_TTL_SECONDS = int(os.environ.get('RESULT_TTL_SECONDS', '1800'))
    PREVIEW_ROWS = 100

    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()


class TestConfig(Config):
    TESTING = True
    DEBUG = False
