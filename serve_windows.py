# Windows production server (waitress)
# Run with:  python serve_windows.py
from waitress import serve
from app import app

serve(app, host='0.0.0.0', port=5000, threads=8)
