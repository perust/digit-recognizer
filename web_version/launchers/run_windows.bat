@REM Created: 2026-01-26
@REM Handwritten Digit Recognition Launcher for Windows
@REM Double-click this file to run the application

@echo off
cd /d "%~dp0.."

echo ==============================================
echo   Handwritten Digit Recognition (Web Version)
echo ==============================================
echo.
echo Starting server...
echo Browser will open automatically.
echo.
echo Press Ctrl+C to stop the server.
echo.

python app_standalone.py

pause
