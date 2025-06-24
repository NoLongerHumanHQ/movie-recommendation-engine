#!/usr/bin/env python3
"""
Script to download sample movie data for the recommendation system.
This will download the TMDB 5000 Movie Dataset and save it to the data directory.
"""

import os
import requests
import pandas as pd
import time
import sys
from src.data_processor import DataProcessor

def download_sample_data():
    """Download sample TMDB movie data and process it"""
    data_dir = "data"
    data_path = os.path.join(data_dir, "movies_metadata.csv")
    processed_path = os.path.join(data_dir, "processed_data.pkl")
    
    print("Starting data download and processing...")
    
    # Create data directory if it doesn't exist
    try:
        os.makedirs(data_dir, exist_ok=True)
        print(f"Data directory created/verified at {data_dir}")
    except Exception as e:
        print(f"Error creating data directory: {e}")
        return False
    
    # Download sample data if it doesn't exist
    if not os.path.exists(data_path):
        print("Downloading sample TMDB dataset...")
        url = "https://raw.githubusercontent.com/Kamal2511/Movie-Recommender-System/main/tmdb_5000_movies.csv"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Download and save
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(data_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Sample data downloaded and saved to {data_path}")
                    break
                else:
                    print(f"Failed to download data: HTTP {response.status_code}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
            except Exception as e:
                print(f"Error downloading sample data (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("All download attempts failed.")
                    return False
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
            return False
    else:
        print(f"Processed data already exists at {processed_path}")
    
    # Verify the data is usable
    try:
        processor = DataProcessor()
        data = processor.load_processed_data(processed_path)
        movies_df = data['movies_df']
        
        if movies_df is None or len(movies_df) == 0:
            print("Processed data appears to be empty or corrupted.")
            return False
            
        print(f"Data verification successful: {len(movies_df)} movies loaded.")
    except Exception as e:
        print(f"Error verifying processed data: {e}")
        return False
    
    print("\nSetup complete! You can now run the app with: streamlit run app.py")
    return True

if __name__ == "__main__":
    success = download_sample_data()
    sys.exit(0 if success else 1) 