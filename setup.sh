#!/bin/bash

pip install -r requirements.txt
pip install -e .

# Create necessary directories
mkdir -p data/raw/nih_kaggle
mkdir -p data/processed/resized
mkdir -p logs/checkpoint
mkdir -p configs

# Set permissions
chmod +x scripts/*.py

python3 ./scripts/download_nih.py
python3 ./scripts/preprocess_label.py
python3 ./scripts/preprocess_resize.py
python3 ./scripts/preprocess_split.py