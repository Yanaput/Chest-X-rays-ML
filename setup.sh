#!/bin/bash

pip install -r requirements.txt
pip install -e .
apt-get update && apt-get install tmux

# Set permissions
chmod +x scripts/*.py

python3 ./scripts/download_nih.py
python3 ./scripts/preprocess_label.py
python3 ./scripts/preprocess_resize.py
python3 ./scripts/preprocess_split.py