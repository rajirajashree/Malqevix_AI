import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = int(os.environ.get('WEB_CONCURRENCY', '1'))
preload_app = True          # load the models once in the master; workers share the memory
timeout = 120
accesslog = '-'
