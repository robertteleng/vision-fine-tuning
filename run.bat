@echo off
title VR Pillar Detector
echo ============================================
echo   VR Pillar Detector
echo ============================================
echo.

:: Check if virtual environment exists
if not exist .venv (
    echo [ERROR] Virtual environment not found.
    echo Please run install.bat first.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment and run
echo Starting application...
echo The web interface will open in your browser.
echo.
echo Press Ctrl+C to stop the server.
echo.
call .venv\Scripts\activate.bat
python app.py

pause
