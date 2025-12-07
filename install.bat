@echo off
title VR Pillar Detector - Installation
echo ============================================
echo   VR Pillar Detector - Installation
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

:: Create virtual environment
echo [1/3] Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)
echo.

:: Activate and install dependencies
echo [2/3] Installing dependencies (this may take a few minutes)...
call .venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo.

:: Verify installation
echo [3/3] Verifying installation...
python -c "import gradio; import ultralytics; print('[OK] All dependencies installed correctly')"
if errorlevel 1 (
    echo [ERROR] Installation verification failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Installation completed successfully!
echo ============================================
echo.
echo To start the application, run: run.bat
echo.
pause
