import requests
import pandas as pd
import os
import time
import json
from PIL import Image
from io import BytesIO
import streamlit as st

def get_poster_url(movie_id, base_url="https://image.tmdb.org/t/p/w500", api_key=None):
    """
    Get the poster URL for a movie from TMDB API
    
    Args:
        movie_id (int): TMDB movie ID
        base_url (str): Base URL for TMDB images
        api_key (str): TMDB API key
        
    Returns:
        str: Full poster URL or None if not found
    """
    if not api_key:
        api_key = st.secrets.get("TMDB_API_KEY", "")
        if not api_key:
            return None
            
    try:
        # Make API request to get movie details
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": api_key}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("poster_path"):
                return f"{base_url}{data['poster_path']}"
            else:
                return None
        else:
            return None
    except:
        return None

def get_poster_image(poster_url, placeholder_path="movie_recommender/data/placeholder.jpg"):
    """
    Get poster image from URL or return placeholder
    
    Args:
        poster_url (str): URL to the poster image
        placeholder_path (str): Path to placeholder image
        
    Returns:
        PIL.Image: Poster image or placeholder
    """
    try:
        if poster_url:
            response = requests.get(poster_url)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            
        # If URL is None or request failed, return placeholder
        if os.path.exists(placeholder_path):
            return Image.open(placeholder_path)
        else:
            # Create a simple gray placeholder with text
            img = Image.new('RGB', (500, 750), color=(50, 50, 50))
            return img
    except:
        # Create a simple gray placeholder
        img = Image.new('RGB', (500, 750), color=(50, 50, 50))
        return img

def format_genres(genres_str):
    """
    Format genres string for display
    
    Args:
        genres_str (str): String with genres
        
    Returns:
        str: Formatted genres for display
    """
    if not genres_str or pd.isna(genres_str):
        return "Unknown"
        
    # Try to parse JSON if in JSON format
    try:
        genres = json.loads(genres_str.replace("'", "\""))
        if isinstance(genres, list):
            genre_names = [g.get('name', '') for g in genres if 'name' in g]
            return ", ".join(genre_names)
    except:
        pass
        
    # If not JSON, just clean the string
    return genres_str.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")

def create_star_rating(rating, max_rating=10, filled_char="★", empty_char="☆"):
    """
    Create a star rating visualization
    
    Args:
        rating (float): Rating value
        max_rating (int): Maximum possible rating
        filled_char (str): Character for filled stars
        empty_char (str): Character for empty stars
        
    Returns:
        str: Star rating visualization
    """
    if pd.isna(rating):
        return "No rating"
        
    # Convert to 5-star scale if needed
    if max_rating != 5:
        rating = (rating / max_rating) * 5
        max_rating = 5
        
    # Round to nearest half star
    rating_rounded = round(rating * 2) / 2
    
    # Create star string
    filled_stars = filled_char * int(rating_rounded)
    half_star = "½" if rating_rounded % 1 == 0.5 else ""
    empty_stars = empty_char * int(max_rating - rating_rounded)
    
    return f"{filled_stars}{half_star}{empty_stars} ({rating:.1f}/10)"

def get_movie_backdrop(movie_id, api_key=None, base_url="https://image.tmdb.org/t/p/w1280"):
    """
    Get movie backdrop image URL from TMDB
    
    Args:
        movie_id (int): TMDB movie ID
        api_key (str): TMDB API key
        base_url (str): Base URL for TMDB images
        
    Returns:
        str: Backdrop image URL or None if not found
    """
    if not api_key:
        api_key = st.secrets.get("TMDB_API_KEY", "")
        if not api_key:
            return None
            
    try:
        # Make API request to get movie details
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": api_key}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("backdrop_path"):
                return f"{base_url}{data['backdrop_path']}"
            else:
                return None
        else:
            return None
    except:
        return None

def create_movie_card(movie, show_poster=True, show_rating=True, on_click=None):
    """
    Create a movie card for Streamlit display
    
    Args:
        movie (pd.Series): Movie data
        show_poster (bool): Whether to show the poster
        show_rating (bool): Whether to show the rating
        on_click (function): Function to call when card is clicked
        
    Returns:
        None: Renders the card directly in Streamlit
    """
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if show_poster and 'poster_path' in movie:
            poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
            st.image(poster_url, use_column_width=True)
        elif show_poster and 'id' in movie:
            poster_url = get_poster_url(movie['id'])
            if poster_url:
                st.image(poster_url, use_column_width=True)
            else:
                st.image("movie_recommender/data/placeholder.jpg", use_column_width=True)
    
    with col2:
        st.subheader(movie['title'])
        
        # Show year if available
        if 'year' in movie and not pd.isna(movie['year']):
            st.write(f"**Year:** {int(movie['year'])}")
        elif 'release_date' in movie and not pd.isna(movie['release_date']):
            try:
                year = movie['release_date'].split('-')[0]
                st.write(f"**Year:** {year}")
            except:
                pass
        
        # Show genres if available
        if 'genres' in movie and not pd.isna(movie['genres']):
            st.write(f"**Genres:** {format_genres(movie['genres'])}")
        
        # Show rating if available and requested
        if show_rating:
            if 'vote_average' in movie and not pd.isna(movie['vote_average']):
                st.write(create_star_rating(movie['vote_average']))
            elif 'rating' in movie and not pd.isna(movie['rating']):
                st.write(create_star_rating(movie['rating']))
        
        # Show overview if available
        if 'overview' in movie and not pd.isna(movie['overview']):
            st.write(movie['overview'][:150] + "..." if len(movie['overview']) > 150 else movie['overview'])
        
        # Add click functionality if provided
        if on_click:
            st.button(f"More about {movie['title']}", key=f"btn_{movie['id']}", on_click=on_click, args=(movie,))

def export_recommendations(recommendations, format="csv"):
    """
    Export recommendations to a file
    
    Args:
        recommendations (pd.DataFrame): Recommendations dataframe
        format (str): Export format (csv or txt)
        
    Returns:
        BytesIO: File buffer for download
    """
    if format == "csv":
        return recommendations.to_csv(index=False).encode('utf-8')
    elif format == "txt":
        text = "Recommended Movies:\n\n"
        for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
            text += f"{i}. {movie['title']}"
            if 'year' in movie and not pd.isna(movie['year']):
                text += f" ({int(movie['year'])})"
            if 'vote_average' in movie and not pd.isna(movie['vote_average']):
                text += f" - Rating: {movie['vote_average']}/10"
            text += "\n"
        return text.encode('utf-8')
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_tmdb_data(file_path):
    """
    Load and process TMDB dataset
    
    Args:
        file_path (str): Path to the TMDB CSV file
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Extract year from release_date
    if 'release_date' in df.columns:
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    # Ensure id column is present
    if 'id' not in df.columns and 'movie_id' in df.columns:
        df['id'] = df['movie_id']
    
    return df

def setup_session_state():
    """
    Initialize session state variables
    
    Returns:
        None
    """
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None
    
    if 'favorite_movies' not in st.session_state:
        st.session_state.favorite_movies = []
    
    if 'favorite_genres' not in st.session_state:
        st.session_state.favorite_genres = []
    
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    
    if 'movie_views' not in st.session_state:
        st.session_state.movie_views = {}
    
    if 'recommendation_clicks' not in st.session_state:
        st.session_state.recommendation_clicks = {}

def track_movie_view(movie_id):
    """
    Track when a movie is viewed
    
    Args:
        movie_id (int): ID of the movie being viewed
        
    Returns:
        None
    """
    if 'movie_views' not in st.session_state:
        st.session_state.movie_views = {}
    
    # Increment view count
    if movie_id in st.session_state.movie_views:
        st.session_state.movie_views[movie_id] += 1
    else:
        st.session_state.movie_views[movie_id] = 1
    
    # Add to recently viewed
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    
    # Remove if already in list
    st.session_state.recently_viewed = [m for m in st.session_state.recently_viewed if m != movie_id]
    
    # Add to front of list
    st.session_state.recently_viewed.insert(0, movie_id)
    
    # Keep only last 10
    st.session_state.recently_viewed = st.session_state.recently_viewed[:10]

def track_recommendation_click(movie_id, recommendation_type):
    """
    Track when a recommendation is clicked
    
    Args:
        movie_id (int): ID of the recommended movie
        recommendation_type (str): Type of recommendation
        
    Returns:
        None
    """
    if 'recommendation_clicks' not in st.session_state:
        st.session_state.recommendation_clicks = {}
    
    key = f"{recommendation_type}_{movie_id}"
    
    if key in st.session_state.recommendation_clicks:
        st.session_state.recommendation_clicks[key] += 1
    else:
        st.session_state.recommendation_clicks[key] = 1

def get_theme_config():
    """
    Get theme configuration for the app
    
    Returns:
        dict: Theme configuration
    """
    # Check if dark theme is enabled in session state
    dark_theme = st.session_state.get('dark_theme', True)
    
    if dark_theme:
        return {
            "bgcolor": "#121212",
            "textcolor": "#FFFFFF",
            "accentcolor": "#FF4B4B",
            "secondarycolor": "#4B4B4B",
            "font": "sans-serif"
        }
    else:
        return {
            "bgcolor": "#FFFFFF",
            "textcolor": "#121212",
            "accentcolor": "#FF4B4B",
            "secondarycolor": "#E6E6E6",
            "font": "sans-serif"
        }
