#!/usr/bin/env python3
"""
Script to download sample movie data for the recommendation system.
This will download the TMDB 5000 Movie Dataset and save it to the data directory.
"""

import os
import requests
import pandas as pd
from src.data_processor import DataProcessor

def download_sample_data():
    """Download sample TMDB movie data and process it"""
    data_dir = "data"
    data_path = os.path.join(data_dir, "movies_metadata.csv")
    processed_path = os.path.join(data_dir, "processed_data.pkl")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download sample data if it doesn't exist
    if not os.path.exists(data_path):
        print("Downloading sample TMDB dataset...")
        url = "https://raw.githubusercontent.com/Kamal2511/Movie-Recommender-System/main/tmdb_5000_movies.csv"
        
        try:
            # Download and save
            df = pd.read_csv(url)
            df.to_csv(data_path, index=False)
            print(f"Sample data downloaded and saved to {data_path}")
        except Exception as e:
            print(f"Error downloading sample data: {e}")
            return
    else:
        print(f"Sample data already exists at {data_path}")
    
    # Process the data if processed data doesn't exist
    if not os.path.exists(processed_path):
        print("Processing data...")
        
        try:
            # Process the data
            processor = DataProcessor(data_path)
            movies_df = processor.load_data()
            movies_df = processor.preprocess_data()
            similarity_matrix = processor.compute_similarity_matrix()
            
            # Save processed data
            processor.save_processed_data(processed_path)
            print(f"Data processed and saved to {processed_path}")
        except Exception as e:
            print(f"Error processing data: {e}")
    else:
        print(f"Processed data already exists at {processed_path}")
    
    print("\nSetup complete! You can now run the app with: streamlit run app.py")

if __name__ == "__main__":
    download_sample_data() 