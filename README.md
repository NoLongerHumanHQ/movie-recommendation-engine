# 🎬 Movie Recommendation System

A lightweight movie recommendation system built with Streamlit that provides personalized movie suggestions based on content similarity, popularity, and user preferences.

## 🚀 Features

- **Multiple recommendation algorithms**:
  - Content-based filtering (using TF-IDF and cosine similarity)
  - Popularity-based recommendations
  - Hybrid approach (combining content and popularity)
  - User preference-based recommendations

- **Intuitive interface**:
  - Movie search with filters (genre, year, rating)
  - Detailed movie information
  - Similar movies suggestions
  - User preferences management
  - Dark/light theme toggle

- **Additional functionality**:
  - Export recommendations as CSV or text
  - Movie poster display
  - Star rating visualization
  - Recently viewed tracking
  - Responsive design

## 📋 Data Source

The application uses the TMDb 5000 Movie Dataset, which includes:
- Movie metadata (title, overview, genres, etc.)
- Ratings and popularity metrics
- Release dates and other information

If no dataset is found locally, the app will automatically download a sample dataset from GitHub.

## 🛠️ Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) If you want to use TMDB API for movie posters:
   - Create an account on [The Movie Database](https://www.themoviedb.org/)
   - Get an API key from your account settings
   - Create a `.streamlit/secrets.toml` file with:
     ```
     TMDB_API_KEY = "your_api_key_here"
     ```

## 🚀 Usage

1. Run the Streamlit app:
   ```
   cd movie_recommender
   streamlit run app.py
   ```

2. Open your browser and go to the URL shown in your terminal (usually http://localhost:8501)

3. Navigate through the app:
   - **Home**: Browse trending, recent, and top-rated movies
   - **Search Movies**: Find movies with filters
   - **Recommendations**: Get personalized recommendations
   - **Movie Details**: View detailed information about a movie
   - **Preferences**: Set your favorite movies and genres

## 🧩 How It Works

1. **Content-based filtering**:
   - Extracts features from movie metadata (overview, genres, keywords)
   - Uses TF-IDF to convert text to numerical vectors
   - Calculates cosine similarity between movies
   - Recommends movies with highest similarity to user's input

2. **Popularity-based**:
   - Ranks movies based on popularity, vote count, and vote average
   - Filters by minimum vote threshold to ensure quality

3. **Hybrid approach**:
   - Combines content similarity and popularity metrics
   - Uses weighted scoring to balance between similarity and popularity

4. **User preferences**:
   - Stores user's favorite movies and genres
   - Generates recommendations based on these preferences

## 📁 Project Structure

```
movie_recommender/
├── app.py                # Main Streamlit application
├── data/                 # Data directory
│   ├── movies_metadata.csv   # Raw movie data
│   └── processed_data.pkl    # Processed data and similarity matrix
├── src/                  # Source code
│   ├── data_processor.py     # Data loading and processing
│   ├── recommender.py        # Recommendation algorithms
│   └── utils.py              # Utility functions
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## 🔧 Customization

- **Dataset**: Replace the default dataset with your own movie data in CSV format.
- **Recommendation parameters**: Adjust the weights in the hybrid recommendation system.
- **UI**: Modify the CSS in the `local_css()` function to change the appearance.

## 📱 Deployment

This app is designed to be lightweight and can be easily deployed on:

- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Using the provided `requirements.txt`
- **Railway/Render**: Works well with their free tiers

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [The Movie Database (TMDb)](https://www.themoviedb.org/) for the movie data
- [Streamlit](https://streamlit.io/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms 