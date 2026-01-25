"""
Handwritten Digit Recognition - Standalone Application
Packaged as executable with PyInstaller
"""

import os
import sys
import numpy as np
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageFilter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import ndimage
import webbrowser
import threading

# Determine base path for bundled files
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    BASE_DIR = sys._MEIPASS
else:
    # Running as script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Global model variable
model = None
MODEL_PATH = os.path.join(BASE_DIR, 'mnist_model.keras')

# HTML template embedded in Python
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwritten Digit Recognition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            text-align: center;
            max-width: 500px;
            width: 100%;
        }

        h1 {
            color: #1a1a2e;
            margin-bottom: 10px;
            font-size: 1.8rem;
        }

        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 0.95rem;
        }

        .canvas-container {
            position: relative;
            display: inline-block;
            margin-bottom: 20px;
        }

        #canvas {
            border: 3px solid #1a1a2e;
            border-radius: 10px;
            cursor: crosshair;
            background: #000;
            touch-action: none;
        }

        .buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 30px;
        }

        button {
            padding: 12px 30px;
            font-size: 1rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        #predictBtn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        #predictBtn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }

        #clearBtn {
            background: #e74c3c;
            color: white;
        }

        #clearBtn:hover {
            background: #c0392b;
            transform: translateY(-2px);
        }

        .result-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
        }

        .result-title {
            font-size: 1rem;
            color: #666;
            margin-bottom: 15px;
        }

        .predicted-digit {
            font-size: 5rem;
            font-weight: bold;
            color: #1a1a2e;
            line-height: 1;
        }

        .confidence {
            font-size: 1.1rem;
            color: #667eea;
            margin-top: 10px;
        }

        .probability-bars {
            margin-top: 20px;
            text-align: left;
        }

        .prob-row {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }

        .prob-label {
            width: 25px;
            font-weight: bold;
            color: #333;
        }

        .prob-bar-container {
            flex: 1;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 0 10px;
        }

        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
            transition: width 0.5s ease;
        }

        .prob-value {
            width: 50px;
            text-align: right;
            font-size: 0.85rem;
            color: #666;
        }

        .instructions {
            background: #fff3cd;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 0.9rem;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Handwritten Digit Recognition</h1>
        <p class="subtitle">Draw a digit (0-9) and let AI recognize it</p>

        <div class="instructions">
            Draw a single digit in the black canvas below using your mouse or touch
        </div>

        <div class="canvas-container">
            <canvas id="canvas" width="280" height="280"></canvas>
        </div>

        <div class="buttons">
            <button id="predictBtn">Recognize</button>
            <button id="clearBtn">Clear</button>
        </div>

        <div class="result-section">
            <div class="result-title">Prediction Result</div>
            <div class="predicted-digit" id="predictedDigit">-</div>
            <div class="confidence" id="confidence">Draw a digit and click Recognize</div>

            <div class="probability-bars" id="probabilityBars">
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        function initCanvas() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 15;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
        }

        initCanvas();

        function initProbabilityBars() {
            const container = document.getElementById('probabilityBars');
            container.innerHTML = '';
            for (let i = 0; i < 10; i++) {
                const row = document.createElement('div');
                row.className = 'prob-row';
                row.innerHTML = `
                    <span class="prob-label">${i}</span>
                    <div class="prob-bar-container">
                        <div class="prob-bar" id="bar-${i}" style="width: 0%"></div>
                    </div>
                    <span class="prob-value" id="prob-${i}">0%</span>
                `;
                container.appendChild(row);
            }
        }

        initProbabilityBars();

        function getCoordinates(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            if (e.touches) {
                return {
                    x: (e.touches[0].clientX - rect.left) * scaleX,
                    y: (e.touches[0].clientY - rect.top) * scaleY
                };
            }
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }

        function startDrawing(e) {
            e.preventDefault();
            isDrawing = true;
            const coords = getCoordinates(e);
            lastX = coords.x;
            lastY = coords.y;
        }

        function draw(e) {
            e.preventDefault();
            if (!isDrawing) return;

            const coords = getCoordinates(e);

            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(coords.x, coords.y);
            ctx.stroke();

            lastX = coords.x;
            lastY = coords.y;
        }

        function stopDrawing(e) {
            e.preventDefault();
            isDrawing = false;
        }

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchmove', draw);
        canvas.addEventListener('touchend', stopDrawing);

        document.getElementById('clearBtn').addEventListener('click', () => {
            initCanvas();
            document.getElementById('predictedDigit').textContent = '-';
            document.getElementById('confidence').textContent = 'Draw a digit and click Recognize';
            initProbabilityBars();
        });

        document.getElementById('predictBtn').addEventListener('click', async () => {
            const imageData = canvas.toDataURL('image/png');

            try {
                document.getElementById('predictedDigit').textContent = '...';
                document.getElementById('confidence').textContent = 'Analyzing...';

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ image: imageData })
                });

                const result = await response.json();

                if (result.success) {
                    document.getElementById('predictedDigit').textContent = result.digit;
                    document.getElementById('confidence').textContent =
                        `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

                    for (let i = 0; i < 10; i++) {
                        const prob = result.probabilities[i];
                        document.getElementById(`bar-${i}`).style.width = `${prob * 100}%`;
                        document.getElementById(`prob-${i}`).textContent = `${(prob * 100).toFixed(1)}%`;
                    }
                } else {
                    document.getElementById('predictedDigit').textContent = '?';
                    document.getElementById('confidence').textContent = 'Error: ' + result.error;
                }
            } catch (error) {
                document.getElementById('predictedDigit').textContent = '?';
                document.getElementById('confidence').textContent = 'Connection error';
                console.error('Error:', error);
            }
        });
    </script>
</body>
</html>
'''


def create_model():
    """Create a CNN model for digit recognition"""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def center_of_mass_shift(img):
    """Shift image so center of mass is at the center - MNIST style"""
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shift_x = np.round(cols / 2.0 - cx).astype(int)
    shift_y = np.round(rows / 2.0 - cy).astype(int)
    shifted = ndimage.shift(img, [shift_y, shift_x], cval=0, mode='constant')
    return shifted


def preprocess_image(image_data):
    """Preprocess the drawn image for prediction - MNIST style centering"""
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)

    image = Image.open(BytesIO(image_bytes))

    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (0, 0, 0))
        background.paste(image, mask=image.split()[3])
        image = background

    image = image.convert('L')
    img_array = np.array(image)

    threshold = 30
    rows = np.any(img_array > threshold, axis=1)
    cols = np.any(img_array > threshold, axis=0)

    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 28, 28, 1), dtype='float32')

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    margin = 20
    y_min = max(0, y_min - margin)
    y_max = min(img_array.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(img_array.shape[1], x_max + margin)

    digit = img_array[y_min:y_max+1, x_min:x_max+1]

    h, w = digit.shape
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    digit = np.pad(digit, ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)),
                   mode='constant', constant_values=0)

    digit_img = Image.fromarray(digit.astype(np.uint8))
    digit_img = digit_img.resize((20, 20), Image.Resampling.LANCZOS)
    digit_img = digit_img.filter(ImageFilter.GaussianBlur(radius=0.5))

    digit_array = np.array(digit_img)

    final_array = np.zeros((28, 28), dtype=np.uint8)
    final_array[4:24, 4:24] = digit_array

    final_array = center_of_mass_shift(final_array.astype(np.float32))
    final_array = final_array.astype('float32') / 255.0
    final_array = np.clip(final_array, 0, 1)
    final_array = final_array.reshape(1, 28, 28, 1)

    return final_array


def load_model():
    """Load the trained model"""
    global model
    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    return model


@app.route('/')
def index():
    """Render the main page"""
    return HTML_TEMPLATE


@app.route('/predict', methods=['POST'])
def predict():
    """Predict the digit from the drawn image"""
    try:
        data = request.get_json()
        image_data = data['image']

        processed_image = preprocess_image(image_data)

        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])

        probabilities = {str(i): float(predictions[0][i]) for i in range(10)}

        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


def open_browser():
    """Open the default browser after a short delay"""
    webbrowser.open('http://127.0.0.1:8080')


if __name__ == '__main__':
    load_model()

    print("\n" + "="*50)
    print("Handwritten Digit Recognition")
    print("Opening browser at http://127.0.0.1:8080")
    print("Press Ctrl+C to quit")
    print("="*50 + "\n")

    # Open browser after server starts
    threading.Timer(1.5, open_browser).start()

    app.run(debug=False, host='127.0.0.1', port=8080)
