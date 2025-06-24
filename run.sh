#!/bin/bash

echo "Starting Movie Recommender App..."

# Check if data exists, if not run download script
if [ ! -f "data/processed_data.pkl" ]; then
    echo "No data found. Running download script first..."
    python download_sample_data.py
fi

# Run the Streamlit app
streamlit run app.py 