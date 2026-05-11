#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate

# Install/refresh deps (idempotent — pip skips already-satisfied packages).
pip install -q -r requirements.txt

python run.py
