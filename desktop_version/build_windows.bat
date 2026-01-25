@REM Created: 2026-01-25 17:50
@REM Windows Build Script for Digit Recognizer
@REM Run this script on a Windows machine to create an executable

@echo off
echo ============================================
echo Building Digit Recognizer for Windows
echo ============================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8+
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install tensorflow pillow scipy pyinstaller

REM Build executable
echo Building executable...
pyinstaller --onefile --windowed --name "DigitRecognizer" --add-data "mnist_model.keras;." --icon=icon.ico app.py

echo.
echo ============================================
echo Build complete!
echo Executable is in: dist\DigitRecognizer.exe
echo ============================================
pause
