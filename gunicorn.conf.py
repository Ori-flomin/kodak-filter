# Gunicorn config — use this on Linux/Mac servers
# Run with:  gunicorn -c gunicorn.conf.py app:app

import os
bind    = f'0.0.0.0:{os.environ.get("PORT", 5000)}'
workers = 2          # Railway hobby: 512 MB RAM — 4 workers OOM-kills on image processing
threads = 2
timeout = 120
keepalive = 5

# Logging
accesslog = '-'      # stdout
errorlog  = '-'
loglevel  = 'info'
