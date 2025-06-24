@echo off
echo Starting Movie Recommender App...

REM Check if data exists, if not run download script
if not exist "data\processed_data.pkl" (
    echo No data found. Running download script first...
    python download_sample_data.py
)

REM Run the Streamlit app
streamlit run app.py

pause 