"""Production-style server that works on Windows, macOS and Linux:  python serve.py"""
import os

from waitress import serve

from app import create_app

if __name__ == '__main__':
    host, port = os.environ.get('HOST', '127.0.0.1'), int(os.environ.get('PORT', '8000'))
    print(f'Serving on http://{host}:{port}  (Ctrl+C to stop)')
    serve(create_app(), host=host, port=port, threads=4, max_request_body_size=int(os.environ.get('MAX_UPLOAD_MB', '20')) * 1024 * 1024)
