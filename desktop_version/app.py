# Created: 2026-01-25 17:48
"""
Handwritten Digit Recognition - Desktop Application
Cross-platform GUI using Tkinter
"""

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import tensorflow as tf
from tensorflow import keras
from scipy import ndimage

# Determine base path
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path - check parent directory if not found locally
MODEL_PATH = os.path.join(BASE_DIR, 'mnist_model.keras')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'mnist_model.keras')


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.resizable(False, False)

        # Set window size and center on screen
        window_width = 500
        window_height = 650
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Configure style
        self.root.configure(bg='#1a1a2e')

        # Load model
        self.model = None
        self.load_model()

        # Drawing state
        self.last_x = None
        self.last_y = None
        self.line_width = 20

        # Create PIL image for drawing (higher resolution for better quality)
        self.canvas_size = 280
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Build UI
        self.create_widgets()

    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = keras.models.load_model(MODEL_PATH)
                print(f"Model loaded from {MODEL_PATH}")
            else:
                messagebox.showerror("Error", f"Model not found at {MODEL_PATH}\nPlease ensure mnist_model.keras exists.")
                sys.exit(1)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            sys.exit(1)

    def create_widgets(self):
        """Create all UI components"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Title
        title_label = tk.Label(
            main_frame,
            text="Handwritten Digit Recognition",
            font=('Segoe UI', 18, 'bold'),
            fg='white',
            bg='#1a1a2e'
        )
        title_label.pack(pady=(0, 5))

        # Subtitle
        subtitle_label = tk.Label(
            main_frame,
            text="Draw a digit (0-9) and click Recognize",
            font=('Segoe UI', 10),
            fg='#888888',
            bg='#1a1a2e'
        )
        subtitle_label.pack(pady=(0, 15))

        # Canvas frame with border
        canvas_frame = tk.Frame(main_frame, bg='white', padx=3, pady=3)
        canvas_frame.pack()

        # Drawing canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='black',
            cursor='crosshair',
            highlightthickness=0
        )
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='#1a1a2e')
        button_frame.pack(pady=15)

        # Custom button using Frame + Label (more reliable on macOS)
        # Recognize button
        self.recognize_btn_frame = tk.Frame(button_frame, bg='#667eea', cursor='hand2')
        self.recognize_btn_frame.pack(side='left', padx=10)
        self.recognize_btn_label = tk.Label(
            self.recognize_btn_frame,
            text="Recognize",
            font=('Segoe UI', 12, 'bold'),
            fg='black',
            bg='#667eea',
            padx=25,
            pady=10,
            cursor='hand2'
        )
        self.recognize_btn_label.pack()

        # Bind click events to both frame and label
        self.recognize_btn_frame.bind('<Button-1>', lambda e: self.recognize_digit())
        self.recognize_btn_label.bind('<Button-1>', lambda e: self.recognize_digit())
        self.recognize_btn_frame.bind('<Enter>', lambda e: self.on_btn_hover(self.recognize_btn_frame, self.recognize_btn_label, '#5563c9'))
        self.recognize_btn_frame.bind('<Leave>', lambda e: self.on_btn_leave(self.recognize_btn_frame, self.recognize_btn_label, '#667eea'))
        self.recognize_btn_label.bind('<Enter>', lambda e: self.on_btn_hover(self.recognize_btn_frame, self.recognize_btn_label, '#5563c9'))
        self.recognize_btn_label.bind('<Leave>', lambda e: self.on_btn_leave(self.recognize_btn_frame, self.recognize_btn_label, '#667eea'))

        # Clear button
        self.clear_btn_frame = tk.Frame(button_frame, bg='#e74c3c', cursor='hand2')
        self.clear_btn_frame.pack(side='left', padx=10)
        self.clear_btn_label = tk.Label(
            self.clear_btn_frame,
            text="Clear",
            font=('Segoe UI', 12, 'bold'),
            fg='black',
            bg='#e74c3c',
            padx=35,
            pady=10,
            cursor='hand2'
        )
        self.clear_btn_label.pack()

        # Bind click events to both frame and label
        self.clear_btn_frame.bind('<Button-1>', lambda e: self.clear_canvas())
        self.clear_btn_label.bind('<Button-1>', lambda e: self.clear_canvas())
        self.clear_btn_frame.bind('<Enter>', lambda e: self.on_btn_hover(self.clear_btn_frame, self.clear_btn_label, '#c0392b'))
        self.clear_btn_frame.bind('<Leave>', lambda e: self.on_btn_leave(self.clear_btn_frame, self.clear_btn_label, '#e74c3c'))
        self.clear_btn_label.bind('<Enter>', lambda e: self.on_btn_hover(self.clear_btn_frame, self.clear_btn_label, '#c0392b'))
        self.clear_btn_label.bind('<Leave>', lambda e: self.on_btn_leave(self.clear_btn_frame, self.clear_btn_label, '#e74c3c'))

        # Result frame
        result_frame = tk.Frame(main_frame, bg='#f8f9fa', padx=20, pady=15)
        result_frame.pack(fill='x', pady=(10, 0))

        # Result title
        result_title = tk.Label(
            result_frame,
            text="Prediction Result",
            font=('Segoe UI', 10),
            fg='#666666',
            bg='#f8f9fa'
        )
        result_title.pack()

        # Predicted digit
        self.digit_label = tk.Label(
            result_frame,
            text="-",
            font=('Segoe UI', 48, 'bold'),
            fg='#1a1a2e',
            bg='#f8f9fa'
        )
        self.digit_label.pack()

        # Confidence
        self.confidence_label = tk.Label(
            result_frame,
            text="Draw a digit and click Recognize",
            font=('Segoe UI', 10),
            fg='#667eea',
            bg='#f8f9fa'
        )
        self.confidence_label.pack(pady=(0, 10))

        # Probability bars frame
        self.prob_frame = tk.Frame(result_frame, bg='#f8f9fa')
        self.prob_frame.pack(fill='x')

        # Create probability bars
        self.prob_bars = []
        self.prob_labels = []
        for i in range(10):
            row_frame = tk.Frame(self.prob_frame, bg='#f8f9fa')
            row_frame.pack(fill='x', pady=1)

            # Digit label
            digit_lbl = tk.Label(
                row_frame,
                text=str(i),
                font=('Segoe UI', 9, 'bold'),
                fg='#333333',
                bg='#f8f9fa',
                width=2
            )
            digit_lbl.pack(side='left')

            # Progress bar container
            bar_container = tk.Frame(row_frame, bg='#e0e0e0', height=12)
            bar_container.pack(side='left', fill='x', expand=True, padx=5)
            bar_container.pack_propagate(False)

            # Progress bar
            bar = tk.Frame(bar_container, bg='#667eea', width=0)
            bar.place(x=0, y=0, relheight=1)
            self.prob_bars.append((bar_container, bar))

            # Percentage label
            pct_lbl = tk.Label(
                row_frame,
                text="0%",
                font=('Segoe UI', 8),
                fg='#666666',
                bg='#f8f9fa',
                width=6
            )
            pct_lbl.pack(side='right')
            self.prob_labels.append(pct_lbl)

    def on_btn_hover(self, frame, label, color):
        """Button hover effect"""
        frame.config(bg=color)
        label.config(bg=color)

    def on_btn_leave(self, frame, label, color):
        """Button leave effect"""
        frame.config(bg=color)
        label.config(bg=color)

    def start_draw(self, event):
        """Start drawing"""
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        """Draw line on canvas"""
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y,
                event.x, event.y,
                fill='white',
                width=self.line_width,
                capstyle=tk.ROUND,
                smooth=True
            )
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255,
                width=self.line_width
            )
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        """Stop drawing"""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.digit_label.config(text="-")
        self.confidence_label.config(text="Draw a digit and click Recognize")

        # Reset probability bars
        for i in range(10):
            container, bar = self.prob_bars[i]
            bar.place(x=0, y=0, relheight=1, width=0)
            self.prob_labels[i].config(text="0%")

    def center_of_mass_shift(self, img):
        """Shift image so center of mass is at the center"""
        cy, cx = ndimage.center_of_mass(img)
        rows, cols = img.shape
        shift_x = np.round(cols / 2.0 - cx).astype(int)
        shift_y = np.round(rows / 2.0 - cy).astype(int)
        shifted = ndimage.shift(img, [shift_y, shift_x], cval=0, mode='constant')
        return shifted

    def preprocess_image(self):
        """Preprocess the drawn image for prediction"""
        # Get image as numpy array
        img_array = np.array(self.image)

        # Find bounding box
        threshold = 30
        rows = np.any(img_array > threshold, axis=1)
        cols = np.any(img_array > threshold, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        # Get bounding box
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Crop with margin
        margin = 20
        y_min = max(0, y_min - margin)
        y_max = min(img_array.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(img_array.shape[1], x_max + margin)

        digit = img_array[y_min:y_max+1, x_min:x_max+1]

        # Make square
        h, w = digit.shape
        max_dim = max(h, w)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        digit = np.pad(digit, ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)),
                       mode='constant', constant_values=0)

        # Resize to 20x20
        digit_img = Image.fromarray(digit.astype(np.uint8))
        digit_img = digit_img.resize((20, 20), Image.Resampling.LANCZOS)
        digit_img = digit_img.filter(ImageFilter.GaussianBlur(radius=0.5))

        digit_array = np.array(digit_img)

        # Create 28x28 image with digit centered
        final_array = np.zeros((28, 28), dtype=np.uint8)
        final_array[4:24, 4:24] = digit_array

        # Apply center of mass shift
        final_array = self.center_of_mass_shift(final_array.astype(np.float32))

        # Normalize
        final_array = final_array.astype('float32') / 255.0
        final_array = np.clip(final_array, 0, 1)
        final_array = final_array.reshape(1, 28, 28, 1)

        return final_array

    def recognize_digit(self):
        """Recognize the drawn digit"""
        # Preprocess image
        processed = self.preprocess_image()

        if processed is None:
            messagebox.showwarning("Warning", "Please draw a digit first!")
            return

        # Predict
        predictions = self.model.predict(processed, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])

        # Update display
        self.digit_label.config(text=str(predicted_digit))
        self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")

        # Update probability bars
        for i in range(10):
            prob = float(predictions[0][i])
            container, bar = self.prob_bars[i]
            container_width = container.winfo_width()
            bar_width = int(container_width * prob)
            bar.place(x=0, y=0, relheight=1, width=bar_width)
            self.prob_labels[i].config(text=f"{prob*100:.1f}%")


def main():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
