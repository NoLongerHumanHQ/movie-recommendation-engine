import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re
from datetime import datetime

class DataProcessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.movies_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        
    def load_data(self, file_path=None):
        """Load movie data from CSV file"""
        if file_path:
            self.data_path = file_path
        
        if not self.data_path:
            raise ValueError("No data path provided")
            
        self.movies_df = pd.read_csv(self.data_path)
        return self.movies_df
    
    def preprocess_data(self):
        """Clean and preprocess the movie data"""
        if self.movies_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Drop duplicates
        self.movies_df.drop_duplicates(subset=['title'], inplace=True)
        
        # Handle missing values
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        
        # Create a combined text feature for content-based filtering
        self.movies_df['combined_features'] = self.movies_df.apply(self._create_combined_features, axis=1)
        
        # Create year column from release_date if available
        if 'release_date' in self.movies_df.columns:
            self.movies_df['year'] = self.movies_df['release_date'].apply(self._extract_year)
        
        return self.movies_df
    
    def _create_combined_features(self, row):
        """Combine relevant features into a single text for TF-IDF processing"""
        combined = ""
        
        # Add genres if available
        if 'genres' in self.movies_df.columns and pd.notna(row['genres']):
            combined += str(row['genres']) + " "
        
        # Add overview if available
        if 'overview' in self.movies_df.columns and pd.notna(row['overview']):
            combined += str(row['overview']) + " "
            
        # Add keywords if available
        if 'keywords' in self.movies_df.columns and pd.notna(row['keywords']):
            combined += str(row['keywords']) + " "
        
        return combined.lower()
    
    def _extract_year(self, date_str):
        """Extract year from date string"""
        if pd.isna(date_str) or date_str == '':
            return np.nan
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').year
        except:
            # Try to extract year with regex
            year_match = re.search(r'(\d{4})', str(date_str))
            if year_match:
                return int(year_match.group(1))
            return np.nan
    
    def compute_similarity_matrix(self):
        """Compute TF-IDF and cosine similarity matrix for content-based filtering"""
        if self.movies_df is None or 'combined_features' not in self.movies_df.columns:
            raise ValueError("Data not properly preprocessed. Call preprocess_data() first.")
        
        # Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Construct the TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['combined_features'])
        
        # Compute the cosine similarity matrix
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        return cosine_sim
    
    def save_processed_data(self, output_path):
        """Save processed data and similarity matrix to disk"""
        if self.movies_df is None:
            raise ValueError("No data processed. Call preprocess_data() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed data as pickle file
        data_to_save = {
            'movies_df': self.movies_df,
            'tfidf_matrix': self.tfidf_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
        print(f"Processed data saved to {output_path}")
    
    def load_processed_data(self, input_path):
        """Load processed data and similarity matrix from disk"""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        self.movies_df = data['movies_df']
        self.tfidf_matrix = data['tfidf_matrix']
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        
        print(f"Processed data loaded from {input_path}")
        return data
    
    def get_movies_by_year_range(self, start_year, end_year):
        """Filter movies by year range"""
        if 'year' not in self.movies_df.columns:
            raise ValueError("Year column not available in dataset")
        
        return self.movies_df[(self.movies_df['year'] >= start_year) & 
                             (self.movies_df['year'] <= end_year)]
    
    def get_movies_by_genre(self, genre):
        """Filter movies by genre"""
        if 'genres' not in self.movies_df.columns:
            raise ValueError("Genres column not available in dataset")
        
        return self.movies_df[self.movies_df['genres'].str.contains(genre, case=False, na=False)]
    
    def get_movies_by_rating(self, min_rating):
        """Filter movies by minimum rating"""
        rating_col = None
        if 'vote_average' in self.movies_df.columns:
            rating_col = 'vote_average'
        elif 'rating' in self.movies_df.columns:
            rating_col = 'rating'
        else:
            raise ValueError("Rating column not available in dataset")
        
        return self.movies_df[self.movies_df[rating_col] >= min_rating]
    
    def get_top_rated_movies(self, n=10):
        """Get top N rated movies"""
        rating_col = None
        if 'vote_average' in self.movies_df.columns:
            rating_col = 'vote_average'
        elif 'rating' in self.movies_df.columns:
            rating_col = 'rating'
        else:
            raise ValueError("Rating column not available in dataset")
        
        # Filter out movies with very few votes to avoid movies with few but high ratings
        vote_col = None
        if 'vote_count' in self.movies_df.columns:
            vote_col = 'vote_count'
            vote_threshold = self.movies_df[vote_col].quantile(0.5)
            filtered_df = self.movies_df[self.movies_df[vote_col] >= vote_threshold]
        else:
            filtered_df = self.movies_df
            
        return filtered_df.sort_values(by=rating_col, ascending=False).head(n)
    
    def get_most_popular_movies(self, n=10):
        """Get most popular movies based on vote count or popularity metric"""
        pop_col = None
        if 'popularity' in self.movies_df.columns:
            pop_col = 'popularity'
        elif 'vote_count' in self.movies_df.columns:
            pop_col = 'vote_count'
        else:
            raise ValueError("Popularity or vote count column not available in dataset")
            
        return self.movies_df.sort_values(by=pop_col, ascending=False).head(n)
    
    def get_most_recent_movies(self, n=10):
        """Get most recent movies"""
        if 'year' not in self.movies_df.columns:
            raise ValueError("Year column not available in dataset")
            
        return self.movies_df.sort_values(by='year', ascending=False).head(n)
    
    def search_movies(self, query):
        """Search for movies by title"""
        if self.movies_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        return self.movies_df[self.movies_df['title'].str.contains(query, case=False, na=False)] 