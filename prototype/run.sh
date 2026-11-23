#!/bin/sh
# Start the Latent Map prototype, then open http://127.0.0.1:8765
cd "$(dirname "$0")" && PYTHONPATH=../spike:../basemap exec ../spike/.venv/bin/python server.py
