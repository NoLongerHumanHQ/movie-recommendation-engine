import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, movies_df=None, similarity_matrix=None):
        self.movies_df = movies_df
        self.similarity_matrix = similarity_matrix
        
    def set_data(self, movies_df, similarity_matrix=None):
        """Set the data for the recommender"""
        self.movies_df = movies_df
        self.similarity_matrix = similarity_matrix
        
    def get_content_based_recommendations(self, movie_title, n=10):
        """Get content-based recommendations similar to given movie"""
        if self.movies_df is None or len(self.movies_df) == 0:
            raise ValueError("No movie data available. Call set_data() first.")
            
        if self.similarity_matrix is None:
            raise ValueError("No similarity matrix available.")
            
        # Get the index of the movie
        idx = self.movies_df[self.movies_df['title'] == movie_title].index
        
        if len(idx) == 0:
            # Try partial match if exact match not found
            similar_titles = self.movies_df[self.movies_df['title'].str.contains(movie_title, case=False)]
            
            if len(similar_titles) > 0:
                print(f"Exact match for '{movie_title}' not found. Using closest match: '{similar_titles.iloc[0]['title']}'")
                idx = similar_titles.index[0]
            else:
                raise ValueError(f"Movie '{movie_title}' not found in the dataset.")
        else:
            idx = idx[0]
            
        # Get similarity scores for the movie
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort movies by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top N similar movies
        recommendations = self.movies_df.iloc[movie_indices].copy()
        
        # Add similarity score to the dataframe
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations
    
    def get_popularity_based_recommendations(self, n=10):
        """Get recommendations based on popularity"""
        if self.movies_df is None or len(self.movies_df) == 0:
            raise ValueError("No movie data available. Call set_data() first.")
            
        # Check for popularity column
        if 'popularity' in self.movies_df.columns:
            return self.movies_df.sort_values('popularity', ascending=False).head(n)
        
        # Try vote count if popularity not available
        elif 'vote_count' in self.movies_df.columns:
            return self.movies_df.sort_values('vote_count', ascending=False).head(n)
            
        # Try vote average if vote count not available
        elif 'vote_average' in self.movies_df.columns:
            # Only consider movies with a minimum number of votes
            vote_counts = self.movies_df['vote_count']
            vote_averages = self.movies_df['vote_average']
            
            # Calculate min votes required to be in the chart (90th percentile)
            min_votes = vote_counts.quantile(0.9)
            
            # Filter out movies with low vote counts
            qualified = self.movies_df[self.movies_df['vote_count'] >= min_votes]
            
            return qualified.sort_values('vote_average', ascending=False).head(n)
        
        else:
            # Fallback to returning some random movies
            return self.movies_df.sample(min(n, len(self.movies_df)))
    
    def get_genre_based_recommendations(self, genre, n=10):
        """Get recommendations based on genre"""
        if self.movies_df is None or len(self.movies_df) == 0:
            raise ValueError("No movie data available. Call set_data() first.")
            
        if 'genres' not in self.movies_df.columns:
            raise ValueError("Genres column not available in dataset")
            
        # Filter movies by genre
        genre_movies = self.movies_df[self.movies_df['genres'].str.contains(genre, case=False, na=False)]
        
        # If there are popularity or vote metrics, sort by them
        if 'popularity' in genre_movies.columns:
            return genre_movies.sort_values('popularity', ascending=False).head(n)
        elif 'vote_average' in genre_movies.columns and 'vote_count' in genre_movies.columns:
            # Only consider movies with a minimum number of votes
            qualified = genre_movies[genre_movies['vote_count'] >= genre_movies['vote_count'].quantile(0.5)]
            return qualified.sort_values('vote_average', ascending=False).head(n)
        else:
            # Return random selection if no popularity metrics
            return genre_movies.sample(min(n, len(genre_movies)))
    
    def get_hybrid_recommendations(self, movie_title, weight_content=0.7, weight_popularity=0.3, n=10):
        """Get hybrid recommendations combining content-based and popularity-based approaches"""
        if self.movies_df is None or len(self.movies_df) == 0:
            raise ValueError("No movie data available. Call set_data() first.")
            
        # Get content-based recommendations
        try:
            content_recs = self.get_content_based_recommendations(movie_title, n=n*2)
        except ValueError:
            # If movie not found, fall back to popularity-based only
            return self.get_popularity_based_recommendations(n=n)
            
        # Normalize similarity scores to 0-1 range
        if 'similarity_score' in content_recs.columns:
            max_sim = content_recs['similarity_score'].max()
            if max_sim > 0:
                content_recs['similarity_score'] = content_recs['similarity_score'] / max_sim
                
        # Get popularity score
        if 'popularity' in self.movies_df.columns:
            # Merge popularity scores
            content_recs = content_recs.merge(
                self.movies_df[['id', 'popularity']], 
                on='id', 
                suffixes=('', '_y')
            )
            
            # Normalize popularity to 0-1 range
            max_pop = self.movies_df['popularity'].max()
            if max_pop > 0:
                content_recs['popularity_norm'] = content_recs['popularity'] / max_pop
            else:
                content_recs['popularity_norm'] = 0
                
            # Calculate hybrid score
            content_recs['hybrid_score'] = (
                weight_content * content_recs['similarity_score'] + 
                weight_popularity * content_recs['popularity_norm']
            )
        elif 'vote_average' in self.movies_df.columns and 'vote_count' in self.movies_df.columns:
            # Alternative popularity metric
            content_recs = content_recs.merge(
                self.movies_df[['id', 'vote_average', 'vote_count']], 
                on='id', 
                suffixes=('', '_y')
            )
            
            # Calculate weighted rating
            vote_counts = content_recs['vote_count']
            vote_averages = content_recs['vote_average']
            mean_votes = vote_counts.mean()
            mean_rating = vote_averages.mean()
            
            # Weighted rating formula (IMDB formula)
            content_recs['weighted_rating'] = (
                (vote_counts / (vote_counts + mean_votes)) * vote_averages + 
                (mean_votes / (vote_counts + mean_votes)) * mean_rating
            )
            
            # Normalize weighted rating
            max_rating = content_recs['weighted_rating'].max()
            if max_rating > 0:
                content_recs['weighted_rating_norm'] = content_recs['weighted_rating'] / max_rating
            else:
                content_recs['weighted_rating_norm'] = 0
                
            # Calculate hybrid score
            content_recs['hybrid_score'] = (
                weight_content * content_recs['similarity_score'] + 
                weight_popularity * content_recs['weighted_rating_norm']
            )
        else:
            # If no popularity metrics, just use content-based
            content_recs['hybrid_score'] = content_recs['similarity_score']
            
        # Sort by hybrid score and return top N
        return content_recs.sort_values('hybrid_score', ascending=False).head(n)
    
    def get_recommendations_for_user(self, favorite_movies, favorite_genres=None, n=10):
        """Get personalized recommendations based on user's favorite movies and genres"""
        if self.movies_df is None or len(self.movies_df) == 0:
            raise ValueError("No movie data available. Call set_data() first.")
            
        # Check if there are any favorite movies or genres
        if not favorite_movies and (not favorite_genres or len(favorite_genres) == 0):
            return self.get_popularity_based_recommendations(n=n)
            
        # Start with an empty dataframe for recommendations
        all_recommendations = pd.DataFrame()
        
        # Get content-based recommendations for each favorite movie
        for movie in favorite_movies:
            try:
                movie_recs = self.get_content_based_recommendations(movie, n=n)
                all_recommendations = pd.concat([all_recommendations, movie_recs])
            except ValueError:
                # Skip if movie not found
                continue
                
        # If favorite genres provided, add genre-based recommendations
        if favorite_genres and len(favorite_genres) > 0:
            for genre in favorite_genres:
                try:
                    genre_recs = self.get_genre_based_recommendations(genre, n=n)
                    all_recommendations = pd.concat([all_recommendations, genre_recs])
                except ValueError:
                    # Skip if genre not found
                    continue
                    
        # If no recommendations found, return popularity-based
        if len(all_recommendations) == 0:
            return self.get_popularity_based_recommendations(n=n)
            
        # Remove duplicates and favorite movies from recommendations
        all_recommendations = all_recommendations.drop_duplicates(subset=['id'])
        all_recommendations = all_recommendations[~all_recommendations['title'].isin(favorite_movies)]
        
        # Sort by combined score (if available) or vote average
        if 'similarity_score' in all_recommendations.columns:
            all_recommendations = all_recommendations.sort_values('similarity_score', ascending=False)
        elif 'vote_average' in all_recommendations.columns:
            all_recommendations = all_recommendations.sort_values('vote_average', ascending=False)
            
        return all_recommendations.head(n)
    
    def get_recent_recommendations(self, year_threshold=None, n=10):
        """Get recommendations for recent movies"""
        if self.movies_df is None or len(self.movies_df) == 0:
            raise ValueError("No movie data available. Call set_data() first.")
            
        # Check if year column exists
        if 'year' not in self.movies_df.columns:
            # If no year column, fall back to popularity
            return self.get_popularity_based_recommendations(n=n)
            
        # If no threshold provided, use the median year as threshold
        if year_threshold is None:
            year_threshold = int(self.movies_df['year'].median())
            
        # Filter recent movies
        recent_movies = self.movies_df[self.movies_df['year'] >= year_threshold]
        
        # If no recent movies found, return some popular ones
        if len(recent_movies) == 0:
            return self.get_popularity_based_recommendations(n=n)
            
        # Sort by popularity or vote average
        if 'popularity' in recent_movies.columns:
            return recent_movies.sort_values('popularity', ascending=False).head(n)
        elif 'vote_average' in recent_movies.columns and 'vote_count' in recent_movies.columns:
            # Only consider movies with a minimum number of votes
            vote_threshold = recent_movies['vote_count'].quantile(0.3)  # Lower threshold for recent movies
            qualified = recent_movies[recent_movies['vote_count'] >= vote_threshold]
            
            # If filtering removed too many movies, use original set
            if len(qualified) < n:
                qualified = recent_movies
                
            return qualified.sort_values('vote_average', ascending=False).head(n)
        else:
            # Return random selection if no popularity metrics
            return recent_movies.sample(min(n, len(recent_movies)))
