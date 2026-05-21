# Gunicorn config — use this on Linux/Mac servers
# Run with:  gunicorn -c gunicorn.conf.py app:app

import os
bind    = f'0.0.0.0:{os.environ.get("PORT", 5000)}'
workers = 4          # good starting point: 2 × CPU cores + 1
threads = 2          # each worker handles 2 concurrent requests
timeout = 120        # face detection can take a few seconds
keepalive = 5

# Logging
accesslog = '-'      # stdout
errorlog  = '-'
loglevel  = 'info'
