import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image
import base64
import requests
from io import BytesIO
import json

# Import custom modules
from src.data_processor import DataProcessor
from src.recommender import MovieRecommender
from src.utils import (
    get_poster_url, get_poster_image, format_genres, 
    create_star_rating, create_movie_card, export_recommendations,
    setup_session_state, track_movie_view, track_recommendation_click,
    get_theme_config, get_movie_backdrop
)

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
setup_session_state()

# Add CSS for styling
def local_css():
    st.markdown("""
    <style>
    .movie-card {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
        background-color: rgba(49, 51, 63, 0.7);
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: scale(1.02);
    }
    .card-img {
        border-radius: 5px;
        width: 100%;
    }
    .card-title {
        font-size: 1.2em;
        font-weight: bold;
    }
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        grid-gap: 15px;
    }
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
    }
    .stButton>button {
        width: 100%;
    }
    .rating {
        color: #FFD700;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Helper functions for app pages
def add_bg_from_url(url):
    """Add background image from URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=50)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{img_str});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def load_data():
    """Load or download dataset"""
    data_path = "data/movies_metadata.csv"
    processed_path = "data/processed_data.pkl"
    
    @st.cache_data
    def get_data():
        # Check if processed data exists
        if os.path.exists(processed_path):
            try:
                processor = DataProcessor()
                data = processor.load_processed_data(processed_path)
                return data['movies_df'], data['tfidf_matrix']
            except Exception as e:
                st.error(f"Error loading processed data: {e}")
        
        # If not, check if raw data exists
        if os.path.exists(data_path):
            try:
                # Process the data
                processor = DataProcessor(data_path)
                movies_df = processor.load_data()
                movies_df = processor.preprocess_data()
                similarity_matrix = processor.compute_similarity_matrix()
                
                # Save processed data
                processor.save_processed_data(processed_path)
                
                return movies_df, similarity_matrix
            except Exception as e:
                st.error(f"Error processing data: {e}")
        
        # If no data available, download sample data
        st.info("No data found. Downloading sample TMDB dataset...")
        
        # Use TMDB 5000 sample (smaller dataset)
        url = "https://raw.githubusercontent.com/Kamal2511/Movie-Recommender-System/main/tmdb_5000_movies.csv"
        try:
            # Download and save
            df = pd.read_csv(url)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
            
            # Process the data
            processor = DataProcessor(data_path)
            movies_df = processor.preprocess_data()
            similarity_matrix = processor.compute_similarity_matrix()
            
            # Save processed data
            processor.save_processed_data(processed_path)
            
            return movies_df, similarity_matrix
        except Exception as e:
            st.error(f"Error downloading and processing data: {e}")
            return None, None
    
    return get_data()

# Load data
movies_df, similarity_matrix = load_data()

# Initialize recommender system
recommender = MovieRecommender(movies_df, similarity_matrix)

# Home page
def show_home_page():
    st.title("🎬 Welcome to Movie Recommender")
    st.write("Discover movies you'll love based on your preferences!")
    
    # Featured movies (top trending)
    st.header("📈 Trending Movies")
    with st.spinner("Loading trending movies..."):
        trending_movies = recommender.get_popularity_based_recommendations(n=6)
        
        # Display in a grid
        cols = st.columns(3)
        for i, (_, movie) in enumerate(trending_movies.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="movie-card">
                    <h3>{movie['title']}</h3>
                """, unsafe_allow_html=True)
                
                # Show poster if available
                if 'poster_path' in movie and movie['poster_path']:
                    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                    st.image(poster_url, use_column_width=True)
                elif 'id' in movie:
                    poster_url = get_poster_url(movie['id'])
                    if poster_url:
                        st.image(poster_url, use_column_width=True)
                
                # Add rating
                if 'vote_average' in movie:
                    st.markdown(f"<div class='rating'>{create_star_rating(movie['vote_average'])}</div>", unsafe_allow_html=True)
                
                # Add button to movie details
                if st.button(f"More about {movie['title']}", key=f"trending_{movie['id']}"):
                    st.session_state.selected_movie = movie
                    st.session_state.page = 'movie_details'
                    st.experimental_rerun()
    
    # Recently released movies
    st.header("🆕 Latest Releases")
    with st.spinner("Loading latest releases..."):
        recent_movies = recommender.get_recent_recommendations(n=6)
        
        # Display in a grid
        cols = st.columns(3)
        for i, (_, movie) in enumerate(recent_movies.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="movie-card">
                    <h3>{movie['title']}</h3>
                """, unsafe_allow_html=True)
                
                # Show poster if available
                if 'poster_path' in movie and movie['poster_path']:
                    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                    st.image(poster_url, use_column_width=True)
                elif 'id' in movie:
                    poster_url = get_poster_url(movie['id'])
                    if poster_url:
                        st.image(poster_url, use_column_width=True)
                
                # Add year
                if 'year' in movie and not pd.isna(movie['year']):
                    st.write(f"**Year:** {int(movie['year'])}")
                
                # Add button to movie details
                if st.button(f"More about {movie['title']}", key=f"recent_{movie['id']}"):
                    st.session_state.selected_movie = movie
                    st.session_state.page = 'movie_details'
                    st.experimental_rerun()
    
    # Top rated movies
    st.header("⭐ Top Rated Movies")
    with st.spinner("Loading top rated movies..."):
        top_rated = recommender.movies_df.sort_values('vote_average', ascending=False).head(6)
        
        # Display in a grid
        cols = st.columns(3)
        for i, (_, movie) in enumerate(top_rated.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="movie-card">
                    <h3>{movie['title']}</h3>
                """, unsafe_allow_html=True)
                
                # Show poster if available
                if 'poster_path' in movie and movie['poster_path']:
                    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                    st.image(poster_url, use_column_width=True)
                elif 'id' in movie:
                    poster_url = get_poster_url(movie['id'])
                    if poster_url:
                        st.image(poster_url, use_column_width=True)
                
                # Add rating
                if 'vote_average' in movie:
                    st.markdown(f"<div class='rating'>{create_star_rating(movie['vote_average'])}</div>", unsafe_allow_html=True)
                
                # Add button to movie details
                if st.button(f"More about {movie['title']}", key=f"toprated_{movie['id']}"):
                    st.session_state.selected_movie = movie
                    st.session_state.page = 'movie_details'
                    st.experimental_rerun()
    
    # Quick access to other pages
    st.markdown("---")
    st.subheader("Quick Access")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Search Movies"):
            st.session_state.page = 'search'
            st.experimental_rerun()
    
    with col2:
        if st.button("🧠 Get Recommendations"):
            st.session_state.page = 'recommendations'
            st.experimental_rerun()
    
    with col3:
        if st.button("⚙️ Set Preferences"):
            st.session_state.page = 'preferences'
            st.experimental_rerun()

# Search page
def show_search_page():
    st.title("🔍 Search Movies")
    
    # Search by title
    search_query = st.text_input("Enter movie title to search:", "")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get all unique genres
        all_genres = []
        if 'genres' in movies_df.columns:
            for genre_list in movies_df['genres'].dropna():
                try:
                    genres = json.loads(genre_list.replace("'", "\""))
                    if isinstance(genres, list):
                        all_genres.extend([g['name'] for g in genres if 'name' in g])
                except:
                    # If not JSON format, try simple string splitting
                    if isinstance(genre_list, str):
                        all_genres.extend([g.strip() for g in genre_list.split(',')])
        
        unique_genres = sorted(list(set(all_genres)))
        selected_genre = st.selectbox("Filter by Genre:", ["All Genres"] + unique_genres)
    
    with col2:
        # Year range
        years = movies_df['year'].dropna().astype(int)
        min_year, max_year = int(years.min()), int(years.max())
        year_range = st.slider("Year Range:", min_year, max_year, (min_year, max_year))
    
    with col3:
        # Rating threshold
        min_rating = st.slider("Minimum Rating:", 0.0, 10.0, 0.0, 0.5)
    
    # Apply search and filters
    if st.button("Search") or search_query:
        with st.spinner("Searching movies..."):
            # Start with title search
            if search_query:
                results = movies_df[movies_df['title'].str.contains(search_query, case=False, na=False)]
            else:
                results = movies_df.copy()
            
            # Apply genre filter
            if selected_genre != "All Genres":
                results = results[results['genres'].str.contains(selected_genre, case=False, na=False)]
            
            # Apply year filter
            results = results[(results['year'] >= year_range[0]) & (results['year'] <= year_range[1])]
            
            # Apply rating filter
            if min_rating > 0:
                results = results[results['vote_average'] >= min_rating]
            
            # Display results
            if len(results) > 0:
                st.write(f"Found {len(results)} movies matching your criteria")
                
                # Display in a scrollable container
                with st.container():
                    for i, (_, movie) in enumerate(results.iterrows()):
                        st.markdown("---")
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            # Show poster if available
                            if 'poster_path' in movie and movie['poster_path']:
                                poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                                st.image(poster_url, width=200)
                            elif 'id' in movie:
                                poster_url = get_poster_url(movie['id'])
                                if poster_url:
                                    st.image(poster_url, width=200)
                        
                        with col2:
                            st.subheader(movie['title'])
                            
                            # Show year
                            if 'year' in movie and not pd.isna(movie['year']):
                                st.write(f"**Year:** {int(movie['year'])}")
                            
                            # Show genres
                            if 'genres' in movie and not pd.isna(movie['genres']):
                                st.write(f"**Genres:** {format_genres(movie['genres'])}")
                            
                            # Show rating
                            if 'vote_average' in movie and not pd.isna(movie['vote_average']):
                                st.markdown(f"<div class='rating'>{create_star_rating(movie['vote_average'])}</div>", unsafe_allow_html=True)
                            
                            # Show overview
                            if 'overview' in movie and not pd.isna(movie['overview']):
                                st.write(movie['overview'][:200] + "..." if len(movie['overview']) > 200 else movie['overview'])
                            
                            # Add button to movie details
                            if st.button(f"More about {movie['title']}", key=f"search_{movie['id']}"):
                                st.session_state.selected_movie = movie
                                st.session_state.page = 'movie_details'
                                st.experimental_rerun()
            else:
                st.warning("No movies found matching your criteria. Try adjusting your search.")

# Recommendations page
def show_recommendations_page():
    st.title("🧠 Get Movie Recommendations")
    
    # Recommendation type
    rec_type = st.radio(
        "Choose recommendation method:",
        ["Content-based", "Hybrid (Content + Popularity)", "Based on your preferences"],
        horizontal=True
    )
    
    if rec_type in ["Content-based", "Hybrid (Content + Popularity)"]:
        # Movie input
        movie_input = st.text_input("Enter a movie you like:", "")
        
        # Weights for hybrid (if selected)
        if rec_type == "Hybrid (Content + Popularity)":
            weight_content = st.slider(
                "Content weight:", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.7, 
                step=0.1,
                help="Higher values give more importance to movie content similarity"
            )
            weight_popularity = 1 - weight_content
        else:
            weight_content = 1.0
            weight_popularity = 0.0
        
        # Number of recommendations
        num_recs = st.slider("Number of recommendations:", 5, 20, 10)
        
        # Get recommendations on button click
        if st.button("Get Recommendations") and movie_input:
            with st.spinner(f"Finding movies similar to '{movie_input}'..."):
                try:
                    if rec_type == "Content-based":
                        recommendations = recommender.get_content_based_recommendations(movie_input, n=num_recs)
                    else:  # Hybrid
                        recommendations = recommender.get_hybrid_recommendations(
                            movie_input, 
                            weight_content=weight_content,
                            weight_popularity=weight_popularity,
                            n=num_recs
                        )
                    
                    # Display recommendations
                    if len(recommendations) > 0:
                        st.success(f"Found {len(recommendations)} movies you might like!")
                        
                        # Display in a grid
                        for i, (_, movie) in enumerate(recommendations.iterrows()):
                            st.markdown("---")
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                # Show poster
                                if 'poster_path' in movie and movie['poster_path']:
                                    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                                    st.image(poster_url, width=200)
                                elif 'id' in movie:
                                    poster_url = get_poster_url(movie['id'])
                                    if poster_url:
                                        st.image(poster_url, width=200)
                            
                            with col2:
                                st.subheader(movie['title'])
                                
                                # Show similarity score
                                if 'similarity_score' in movie:
                                    st.write(f"**Similarity:** {movie['similarity_score']:.2f}")
                                elif 'hybrid_score' in movie:
                                    st.write(f"**Match Score:** {movie['hybrid_score']:.2f}")
                                
                                # Show year
                                if 'year' in movie and not pd.isna(movie['year']):
                                    st.write(f"**Year:** {int(movie['year'])}")
                                
                                # Show genres
                                if 'genres' in movie and not pd.isna(movie['genres']):
                                    st.write(f"**Genres:** {format_genres(movie['genres'])}")
                                
                                # Show rating
                                if 'vote_average' in movie and not pd.isna(movie['vote_average']):
                                    st.markdown(f"<div class='rating'>{create_star_rating(movie['vote_average'])}</div>", unsafe_allow_html=True)
                                
                                # Show overview
                                if 'overview' in movie and not pd.isna(movie['overview']):
                                    st.write(movie['overview'][:200] + "..." if len(movie['overview']) > 200 else movie['overview'])
                                
                                # Add button to movie details
                                if st.button(f"More about {movie['title']}", key=f"rec_{movie['id']}"):
                                    st.session_state.selected_movie = movie
                                    st.session_state.page = 'movie_details'
                                    track_recommendation_click(movie['id'], rec_type.lower())
                                    st.experimental_rerun()
                        
                        # Export recommendations
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Export recommendations as CSV"):
                                csv = export_recommendations(recommendations, format="csv")
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name="movie_recommendations.csv",
                                    mime="text/csv"
                                )
                        with col2:
                            if st.button("Export recommendations as Text"):
                                txt = export_recommendations(recommendations, format="txt")
                                st.download_button(
                                    label="Download Text",
                                    data=txt,
                                    file_name="movie_recommendations.txt",
                                    mime="text/plain"
                                )
                    else:
                        st.warning("No recommendations found. Try a different movie.")
                except ValueError as e:
                    st.error(str(e))
    
    else:  # Based on user preferences
        # Check if user has set preferences
        if len(st.session_state.favorite_movies) == 0 and len(st.session_state.favorite_genres) == 0:
            st.warning("You haven't set any preferences yet. Please go to the Preferences page to set your favorite movies and genres.")
            if st.button("Go to Preferences"):
                st.session_state.page = 'preferences'
                st.experimental_rerun()
        else:
            # Show current preferences
            st.subheader("Your current preferences:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Favorite Movies:**")
                if len(st.session_state.favorite_movies) > 0:
                    for movie in st.session_state.favorite_movies:
                        st.write(f"- {movie}")
                else:
                    st.write("No favorite movies set")
            
            with col2:
                st.write("**Favorite Genres:**")
                if len(st.session_state.favorite_genres) > 0:
                    for genre in st.session_state.favorite_genres:
                        st.write(f"- {genre}")
                else:
                    st.write("No favorite genres set")
            
            # Number of recommendations
            num_recs = st.slider("Number of recommendations:", 5, 20, 10)
            
            # Get recommendations on button click
            if st.button("Get Personalized Recommendations"):
                with st.spinner("Finding movies based on your preferences..."):
                    try:
                        recommendations = recommender.get_recommendations_for_user(
                            st.session_state.favorite_movies,
                            st.session_state.favorite_genres,
                            n=num_recs
                        )
                        
                        # Display recommendations
                        if len(recommendations) > 0:
                            st.success(f"Found {len(recommendations)} movies you might like!")
                            
                            # Display in a grid
                            for i, (_, movie) in enumerate(recommendations.iterrows()):
                                st.markdown("---")
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    # Show poster
                                    if 'poster_path' in movie and movie['poster_path']:
                                        poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                                        st.image(poster_url, width=200)
                                    elif 'id' in movie:
                                        poster_url = get_poster_url(movie['id'])
                                        if poster_url:
                                            st.image(poster_url, width=200)
                                
                                with col2:
                                    st.subheader(movie['title'])
                                    
                                    # Show similarity score
                                    if 'similarity_score' in movie:
                                        st.write(f"**Similarity:** {movie['similarity_score']:.2f}")
                                    
                                    # Show year
                                    if 'year' in movie and not pd.isna(movie['year']):
                                        st.write(f"**Year:** {int(movie['year'])}")
                                    
                                    # Show genres
                                    if 'genres' in movie and not pd.isna(movie['genres']):
                                        st.write(f"**Genres:** {format_genres(movie['genres'])}")
                                    
                                    # Show rating
                                    if 'vote_average' in movie and not pd.isna(movie['vote_average']):
                                        st.markdown(f"<div class='rating'>{create_star_rating(movie['vote_average'])}</div>", unsafe_allow_html=True)
                                    
                                    # Show overview
                                    if 'overview' in movie and not pd.isna(movie['overview']):
                                        st.write(movie['overview'][:200] + "..." if len(movie['overview']) > 200 else movie['overview'])
                                    
                                    # Add button to movie details
                                    if st.button(f"More about {movie['title']}", key=f"user_rec_{movie['id']}"):
                                        st.session_state.selected_movie = movie
                                        st.session_state.page = 'movie_details'
                                        track_recommendation_click(movie['id'], 'user_preferences')
                                        st.experimental_rerun()
                            
                            # Export recommendations
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Export recommendations as CSV"):
                                    csv = export_recommendations(recommendations, format="csv")
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name="movie_recommendations.csv",
                                        mime="text/csv"
                                    )
                            with col2:
                                if st.button("Export recommendations as Text"):
                                    txt = export_recommendations(recommendations, format="txt")
                                    st.download_button(
                                        label="Download Text",
                                        data=txt,
                                        file_name="movie_recommendations.txt",
                                        mime="text/plain"
                                    )
                        else:
                            st.warning("No recommendations found based on your preferences. Try adding more movies or genres.")
                    except Exception as e:
                        st.error(f"Error getting recommendations: {str(e)}")

# Movie details page
def show_movie_details_page():
    # Check if a movie is selected
    if st.session_state.selected_movie is None:
        st.warning("No movie selected. Please select a movie first.")
        if st.button("Go to Home"):
            st.session_state.page = 'home'
            st.experimental_rerun()
        return
    
    movie = st.session_state.selected_movie
    
    # Track view
    if 'id' in movie:
        track_movie_view(movie['id'])
    
    # Get backdrop for the movie
    backdrop_url = None
    if 'id' in movie:
        backdrop_url = get_movie_backdrop(movie['id'])
    
    # If backdrop available, use it as background
    if backdrop_url:
        add_bg_from_url(backdrop_url)
    
    # Movie title and year
    st.title(movie['title'])
    if 'year' in movie and not pd.isna(movie['year']):
        st.subheader(f"({int(movie['year'])})")
    
    # Movie details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Show poster
        if 'poster_path' in movie and movie['poster_path']:
            poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
            st.image(poster_url, width=300)
        elif 'id' in movie:
            poster_url = get_poster_url(movie['id'])
            if poster_url:
                st.image(poster_url, width=300)
        
        # Rating
        if 'vote_average' in movie and not pd.isna(movie['vote_average']):
            st.markdown(f"<div class='rating' style='font-size: 1.5em;'>{create_star_rating(movie['vote_average'])}</div>", unsafe_allow_html=True)
        
        # Votes
        if 'vote_count' in movie and not pd.isna(movie['vote_count']):
            st.write(f"**Votes:** {int(movie['vote_count'])}")
        
        # Add to favorites button
        if movie['title'] in st.session_state.favorite_movies:
            if st.button("❤️ Remove from Favorites"):
                st.session_state.favorite_movies.remove(movie['title'])
                st.success(f"Removed '{movie['title']}' from your favorites")
                st.experimental_rerun()
        else:
            if st.button("🤍 Add to Favorites"):
                st.session_state.favorite_movies.append(movie['title'])
                st.success(f"Added '{movie['title']}' to your favorites")
                st.experimental_rerun()
    
    with col2:
        # Tagline
        if 'tagline' in movie and not pd.isna(movie['tagline']) and movie['tagline'] != '':
            st.markdown(f"*{movie['tagline']}*")
        
        # Overview
        if 'overview' in movie and not pd.isna(movie['overview']):
            st.subheader("Overview")
            st.write(movie['overview'])
        
        # Genres
        if 'genres' in movie and not pd.isna(movie['genres']):
            st.subheader("Genres")
            st.write(format_genres(movie['genres']))
        
        # Additional info
        st.subheader("Additional Information")
        
        # Two columns for additional info
        col1, col2 = st.columns(2)
        
        with col1:
            # Runtime
            if 'runtime' in movie and not pd.isna(movie['runtime']):
                st.write(f"**Runtime:** {movie['runtime']} minutes")
            
            # Release date
            if 'release_date' in movie and not pd.isna(movie['release_date']):
                st.write(f"**Release Date:** {movie['release_date']}")
            
            # Budget
            if 'budget' in movie and not pd.isna(movie['budget']) and movie['budget'] > 0:
                st.write(f"**Budget:** ${movie['budget']:,}")
        
        with col2:
            # Revenue
            if 'revenue' in movie and not pd.isna(movie['revenue']) and movie['revenue'] > 0:
                st.write(f"**Revenue:** ${movie['revenue']:,}")
            
            # Popularity
            if 'popularity' in movie and not pd.isna(movie['popularity']):
                st.write(f"**Popularity:** {movie['popularity']:.1f}")
            
            # Original language
            if 'original_language' in movie and not pd.isna(movie['original_language']):
                st.write(f"**Original Language:** {movie['original_language'].upper()}")
    
    # Similar Movies
    st.markdown("---")
    st.subheader("Similar Movies")
    
    with st.spinner("Finding similar movies..."):
        try:
            similar_movies = recommender.get_content_based_recommendations(movie['title'], n=6)
            
            # Display in a grid
            cols = st.columns(3)
            for i, (_, sim_movie) in enumerate(similar_movies.iterrows()):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <h3>{sim_movie['title']}</h3>
                    """, unsafe_allow_html=True)
                    
                    # Show poster
                    if 'poster_path' in sim_movie and sim_movie['poster_path']:
                        poster_url = f"https://image.tmdb.org/t/p/w500{sim_movie['poster_path']}"
                        st.image(poster_url, use_column_width=True)
                    elif 'id' in sim_movie:
                        poster_url = get_poster_url(sim_movie['id'])
                        if poster_url:
                            st.image(poster_url, use_column_width=True)
                    
                    # Add rating
                    if 'vote_average' in sim_movie:
                        st.markdown(f"<div class='rating'>{create_star_rating(sim_movie['vote_average'])}</div>", unsafe_allow_html=True)
                    
                    # Add similarity score
                    if 'similarity_score' in sim_movie:
                        st.write(f"**Similarity:** {sim_movie['similarity_score']:.2f}")
                    
                    # Add button to movie details
                    if st.button(f"More about {sim_movie['title']}", key=f"similar_{sim_movie['id']}"):
                        st.session_state.selected_movie = sim_movie
                        st.experimental_rerun()
        except Exception as e:
            st.error(f"Error finding similar movies: {str(e)}")
            
    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = 'home'
            st.experimental_rerun()
    
    with col2:
        if st.button("Get More Recommendations →"):
            st.session_state.page = 'recommendations'
            st.experimental_rerun()

# Preferences page
def show_preferences_page():
    st.title("⚙️ Preferences")
    st.write("Set your movie preferences to get personalized recommendations")
    
    # Favorite Movies
    st.subheader("Your Favorite Movies")
    
    # Display current favorites
    if len(st.session_state.favorite_movies) > 0:
        st.write("Your current favorite movies:")
        for i, movie in enumerate(st.session_state.favorite_movies):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i+1}. {movie}")
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.favorite_movies.pop(i)
                    st.experimental_rerun()
    else:
        st.write("You haven't added any favorite movies yet.")
    
    # Add new favorite movie
    st.markdown("---")
    st.write("Add a new favorite movie:")
    
    new_movie = st.text_input("Enter movie title:", key="new_favorite_movie")
    
    # Autocomplete suggestions
    if new_movie:
        suggestions = movies_df[movies_df['title'].str.contains(new_movie, case=False)]['title'].head(5).tolist()
        if suggestions:
            selected_suggestion = st.selectbox("Select from suggestions:", [""] + suggestions)
            if selected_suggestion:
                new_movie = selected_suggestion
    
    if st.button("Add to Favorites") and new_movie:
        # Check if movie exists in dataset
        if new_movie in movies_df['title'].values:
            if new_movie not in st.session_state.favorite_movies:
                st.session_state.favorite_movies.append(new_movie)
                st.success(f"Added '{new_movie}' to your favorites")
                st.experimental_rerun()
            else:
                st.warning(f"'{new_movie}' is already in your favorites")
        else:
            st.error(f"Movie '{new_movie}' not found in our database")
    
    # Favorite Genres
    st.markdown("---")
    st.subheader("Your Favorite Genres")
    
    # Get all unique genres
    all_genres = []
    if 'genres' in movies_df.columns:
        for genre_list in movies_df['genres'].dropna():
            try:
                genres = json.loads(genre_list.replace("'", "\""))
                if isinstance(genres, list):
                    all_genres.extend([g['name'] for g in genres if 'name' in g])
            except:
                # If not JSON format, try simple string splitting
                if isinstance(genre_list, str):
                    all_genres.extend([g.strip() for g in genre_list.split(',')])
    
    unique_genres = sorted(list(set(all_genres)))
    
    # Display current favorite genres
    if len(st.session_state.favorite_genres) > 0:
        st.write("Your current favorite genres:")
        for i, genre in enumerate(st.session_state.favorite_genres):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i+1}. {genre}")
            with col2:
                if st.button("Remove", key=f"remove_genre_{i}"):
                    st.session_state.favorite_genres.pop(i)
                    st.experimental_rerun()
    else:
        st.write("You haven't added any favorite genres yet.")
    
    # Add new favorite genre
    st.write("Add a new favorite genre:")
    
    new_genre = st.selectbox("Select genre:", [""] + unique_genres)
    
    if st.button("Add Genre") and new_genre:
        if new_genre not in st.session_state.favorite_genres:
            st.session_state.favorite_genres.append(new_genre)
            st.success(f"Added '{new_genre}' to your favorite genres")
            st.experimental_rerun()
        else:
            st.warning(f"'{new_genre}' is already in your favorite genres")
    
    # Theme preferences
    st.markdown("---")
    st.subheader("App Appearance")
    
    theme = st.checkbox("Dark Theme", value=st.session_state.get('dark_theme', True))
    st.session_state.dark_theme = theme
    
    # Get recommendations based on preferences
    st.markdown("---")
    if len(st.session_state.favorite_movies) > 0 or len(st.session_state.favorite_genres) > 0:
        if st.button("Get Recommendations Based on Preferences"):
            st.session_state.page = 'recommendations'
            st.experimental_rerun()

# Main app layout and functionality
def main():
    st.sidebar.title("🎬 Movie Recommender")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Search Movies", "Recommendations", "Movie Details", "Preferences"]
    )
    
    # Background image
    bg_url = "https://wallpaperaccess.com/full/3658597.jpg"
    add_bg_from_url(bg_url)
    
    # App pages
    if page == "Home":
        show_home_page()
    elif page == "Search Movies":
        show_search_page()
    elif page == "Recommendations":
        show_recommendations_page()
    elif page == "Movie Details":
        show_movie_details_page()
    elif page == "Preferences":
        show_preferences_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app uses TMDB data and APIs to provide movie recommendations "
        "based on content similarity and popularity."
    )
    
    # Dark/Light theme toggle
    theme = st.sidebar.checkbox("Dark Theme", value=True)
    st.session_state.dark_theme = theme

if __name__ == "__main__":
    main() 