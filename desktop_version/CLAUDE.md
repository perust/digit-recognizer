<!-- Created: 2026-01-25 17:45 -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Desktop application for handwritten digit recognition. Standalone GUI application that runs natively on macOS/Windows/Linux.

## Code Style Rules

### File Headers
All new files must include creation date/time as a comment at the top.

## Commands

### Run the Desktop App
```bash
python3 app.py
```

### Build Executable (PyInstaller)
```bash
pyinstaller --onefile --windowed --name "DigitRecognizer" --add-data "mnist_model.keras:." app.py
```

### Install Dependencies
```bash
pip3 install tensorflow pillow scipy tkinter
# or for PyQt version:
pip3 install tensorflow pillow scipy PyQt6
```

## Architecture

### Tech Stack
- **GUI Framework**: Tkinter (built-in) or PyQt6
- **ML Model**: TensorFlow/Keras CNN
- **Packaging**: PyInstaller for standalone executables

### Key Components
- **Canvas**: Native drawing widget for digit input
- **Model**: CNN trained on MNIST (same as web version)
- **Preprocessing**: Same pipeline as web version (center-of-mass shift, etc.)

### Target Platforms
- macOS (.app bundle)
- Windows (.exe)
- Linux (AppImage or binary)

## Files
- `app.py` - Main desktop application
- `mnist_model.keras` - Trained model file
- `mnist.npz` - Local MNIST dataset cache
