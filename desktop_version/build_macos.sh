#!/bin/bash
# Created: 2026-01-25 17:50
# macOS Build Script for Digit Recognizer

echo "============================================"
echo "Building Digit Recognizer for macOS"
echo "============================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed! Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install tensorflow pillow scipy pyinstaller

# Build app bundle
echo "Building app bundle..."
pyinstaller --onedir --windowed --name "DigitRecognizer" \
    --add-data "mnist_model.keras:." \
    --osx-bundle-identifier "com.digitrecognizer.app" \
    app.py

echo ""
echo "============================================"
echo "Build complete!"
echo "App bundle is in: dist/DigitRecognizer.app"
echo "============================================"
