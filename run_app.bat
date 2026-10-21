@echo off
title COGAIT-VISION Clinical AI Dashboard
echo ========================================================
echo Starting COGAIT-VISION Medical Screening Dashboard...
echo ========================================================
echo.
python -m streamlit run app.py --server.port 8501
pause
