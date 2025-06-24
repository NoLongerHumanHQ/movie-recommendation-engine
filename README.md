# Movie Recommendation Engine

A Streamlit-based application that provides personalized movie recommendations using content-based and hybrid filtering techniques.

## Features

- Content-based movie recommendations
- Popularity-based recommendations
- Hybrid recommendation algorithms
- Movie search with filters (genre, year, rating)
- Personalized recommendations based on user preferences
- Movie details page with similar movie suggestions
- User preferences management

## Setup and Installation

### Prerequisites

- Python 3.10 or newer
- Pip package manager

### Local Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd movie-recommendation-engine
   ```

2. Run the setup script to install dependencies and download sample data:
   ```
   # On Windows
   python -m pip install -r requirements.txt
   python download_sample_data.py

   # On macOS/Linux
   chmod +x setup.sh
   ./setup.sh
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

### Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account

2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)

3. Create a new app and point to your GitHub repository

4. Set the main file path to `app.py`

5. Deploy the app

## Troubleshooting

### Common Issues

#### 1. Data Loading Issues

If you encounter data loading issues:

- Make sure the `data` directory exists in your project
- Run `python download_sample_data.py` manually to download and process the sample data
- Check if the processed data file `data/processed_data.pkl` is created successfully

#### 2. AttributeError: 'NoneType' object has no attribute 'columns'

This error occurs when the DataFrame is None because data loading failed. Solutions:

- Delete the `data` directory and run `python download_sample_data.py` again
- Check if the TMDB data URL is accessible
- Make sure you have write permissions in the project directory

#### 3. Deployment Issues on Streamlit Cloud

If deployment fails on Streamlit Cloud:

- Check the `runtime.txt` file to ensure it specifies a supported Python version (e.g., `python-3.10`)
- Verify that `packages.txt` includes necessary system dependencies
- Downgrade package versions in `requirements.txt` if there are compatibility issues
- Ensure the app is set to eagerly load data at startup

## Data Sources

This application uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata) for movie information and recommendations.

## Project Structure

```
movie-recommendation-engine/
├── app.py                 # Main Streamlit application
├── download_sample_data.py # Script to download and process sample data
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version for deployment
├── packages.txt           # System dependencies for deployment
├── setup.py               # Package setup file
├── setup.sh               # Setup script
├── data/                  # Data directory
│   └── processed_data.pkl # Processed movie data
└── src/                   # Source code
    ├── data_processor.py  # Data processing module
    ├── recommender.py     # Recommendation algorithms
    └── utils.py           # Utility functions
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [TMDB](https://www.themoviedb.org/) for their movie database
- [Streamlit](https://streamlit.io/) for their amazing framework 