#!/bin/bash
# Handwritten Digit Recognition Launcher
# Double-click this file to run the application

cd "$(dirname "$0")"

echo "=============================================="
echo "  Handwritten Digit Recognition"
echo "=============================================="
echo ""
echo "Starting server..."
echo "Browser will open automatically."
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

python3 app_standalone.py
