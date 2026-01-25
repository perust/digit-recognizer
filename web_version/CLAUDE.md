<!-- Created: 2026-01-25 17:45 -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Web-based handwritten digit recognition application. Users draw digits in a browser canvas and receive real-time AI predictions.

## Code Style Rules

### File Headers
All new files must include creation date/time as a comment at the top.

## Commands

### Run the Web Server
```bash
python3 app.py
# Opens at http://127.0.0.1:8080
```

### Install Dependencies
```bash
pip3 install tensorflow flask pillow scipy
```

## Architecture

### Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML5 Canvas + Vanilla JavaScript (embedded in Python)
- **ML Model**: TensorFlow/Keras CNN

### API Endpoints
- `GET /` - Serves the drawing canvas UI
- `POST /predict` - Accepts base64 PNG image, returns JSON with digit prediction and probabilities

### Key Components
- **Model**: CNN trained on MNIST (28x28 grayscale images)
- **Preprocessing**: Bounding box extraction → square padding → resize to 20x20 → center in 28x28 → center-of-mass shift
- **Training**: Data augmentation with rotation, zoom, translation (10 epochs)

## Files
- `app.py` - Main Flask application with embedded HTML template
- `mnist_model.keras` - Trained model file
- `mnist.npz` - Local MNIST dataset cache
