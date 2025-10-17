#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Prefer python from venv if present
if [ -f .venv/bin/python ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

"$PY" launcher.py


