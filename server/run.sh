#!/bin/sh
# Start the Latent Map server, then open http://127.0.0.1:8765
cd "$(dirname "$0")" && PYTHONPATH=.:../spike:../basemap:../prototype exec ../spike/.venv/bin/python -m lmserver.app "$@"
