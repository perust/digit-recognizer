# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style Rules

### File Headers
All new files must include creation date/time as a comment at the top:
```python
# Created: 2026-01-25 17:42
```
```javascript
// Created: 2026-01-25 17:42
```
```html
<!-- Created: 2026-01-25 17:42 -->
```

## Project Overview

Handwritten digit recognition web application using a CNN model trained on MNIST dataset. The application provides a browser-based canvas for drawing digits (0-9) and returns real-time predictions with confidence scores.

## Commands

### Run the Application
```bash
python3 app.py
# Opens at http://127.0.0.1:8080
```

### Run Standalone Version (auto-opens browser)
```bash
python3 app_standalone.py
```

### Retrain the Model
Delete `mnist_model.keras` and run the app - it will automatically train a new model with data augmentation (10 epochs).

### Install Dependencies
```bash
pip3 install tensorflow flask pillow scipy
```

## Architecture

### Model Pipeline
1. **CNN Architecture** (`create_model`): Input(28x28x1) → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dropout(0.5) → Dense(128) → Dense(10, softmax)
2. **Training** (`train_model`): Uses data augmentation (rotation ±10%, zoom ±10%, translation ±10%) for 10 epochs
3. **Model Storage**: Saved as `mnist_model.keras`, loaded on startup if exists

### Image Preprocessing (`preprocess_image`)
Critical for matching MNIST format:
1. Decode base64 PNG from canvas
2. Handle RGBA → grayscale conversion
3. Find bounding box of digit, crop with margin
4. Pad to square, resize to 20x20
5. Apply Gaussian blur (radius 0.5)
6. Place in center of 28x28 image (4px padding)
7. Apply center-of-mass shift (`center_of_mass_shift`) - key MNIST preprocessing step
8. Normalize to [0, 1]

### Web Interface
- Single-file architecture with embedded HTML/CSS/JS template
- Flask routes: `/` (canvas UI), `/predict` (POST, returns JSON with digit, confidence, probabilities)
- Canvas: 280x280px, white strokes on black background, line width 15px

### Files
- `app.py`: Development version with training capability
- `app_standalone.py`: Standalone version for distribution (auto-opens browser, no training)
- `mnist_model.keras`: Trained model (~2.7MB)
- `mnist.npz`: Local MNIST dataset cache (avoids SSL issues)
