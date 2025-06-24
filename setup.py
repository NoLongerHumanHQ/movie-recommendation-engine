#!/usr/bin/env python3
"""
Setup script for the Movie Recommendation System.
This script:
1. Downloads sample movie data if not already present
2. Processes the data for the recommendation system
3. Generates a placeholder image for movies without posters
"""

import os
from src.generate_placeholder import generate_placeholder
from download_sample_data import download_sample_data

def setup():
    """Run the complete setup process"""
    print("Setting up Movie Recommendation System...")
    print("-" * 50)
    
    # Create directories
    for directory in ["data", ".streamlit"]:
        os.makedirs(directory, exist_ok=True)
    
    # Generate placeholder image
    print("\nGenerating placeholder image...")
    generate_placeholder()
    
    # Download and process sample data
    print("\nSetting up movie data...")
    download_sample_data()
    
    print("\n" + "-" * 50)
    print("Setup complete! You can now run the app with: streamlit run app.py")

if __name__ == "__main__":
    setup() 