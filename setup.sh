#!/bin/bash

echo "Setting up Movie Recommender App..."

# Create data directory if it doesn't exist
mkdir -p data

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download sample data
echo "Downloading sample data..."
python download_sample_data.py

echo "Setup complete! You can now run the app with: streamlit run app.py" 