"""A drawing board that recognises the digit you write on it.

    python3 -m digit_recognizer.app

Draw with the mouse; the prediction refreshes shortly after you stop moving.
Press "c" or the Clear button to start over.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .preprocess import IMAGE_SIZE, preprocess
from .train import DEFAULT_MODEL_PATH

CANVAS_SIZE = 300  # side of the drawing board, in screen pixels
PEN_WIDTH = 22  # thick enough to survive the 10x downscale to 28x28
PREVIEW_SCALE = 4  # magnification of the 28x28 image fed to the model
PREDICT_DELAY_MS = 150  # idle time before the prediction is refreshed

BACKGROUND = "#1b1d21"
PANEL = "#25282e"
TRACK = "#32363d"
INK = "#ffffff"
ACCENT = "#5b9dff"
MUTED = "#8b909a"

ROW_HEIGHT = 22
BAR_LEFT, BAR_RIGHT = 30, 196


class DigitRecognizerApp(tk.Tk):
    """Tk window holding the canvas, the probability bars and the input preview."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.configure(bg=BACKGROUND)
        self.resizable(False, False)

        self.model = self._load_model(model_path)

        # The strokes are mirrored onto this PIL image as they are drawn.
        # Reading pixels back out of a Tk canvas is platform dependent, so
        # keeping our own copy is both simpler and portable.
        self._image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self._draw = ImageDraw.Draw(self._image)
        self._last_point: tuple[int, int] | None = None
        self._pending_prediction: str | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None

        self._build_layout()
        self._clear()
        self._bring_to_front()

    def _bring_to_front(self) -> None:
        """Raise the board above whatever the user was looking at.

        Started from Finder rather than a shell the process has no claim on the
        foreground, so without this the window opens behind the current app.
        The topmost flag is dropped again straight away, otherwise the board
        would float over everything for the rest of the session.
        """
        self.lift()
        self.attributes("-topmost", True)
        self.after(400, lambda: self.attributes("-topmost", False))
        self.focus_force()

    def _load_model(self, model_path: Path):
        if not model_path.exists():
            messagebox.showerror(
                "Model not found",
                f"No trained model at:\n{model_path}\n\n"
                "Train one first:\n    python3 -m digit_recognizer.train",
            )
            raise SystemExit(1)
        from tensorflow import keras

        return keras.models.load_model(str(model_path))

    # ---------------------------------------------------------------- layout

    def _build_layout(self) -> None:
        root = tk.Frame(self, bg=BACKGROUND, padx=18, pady=16)
        root.pack()

        tk.Label(
            root,
            text="Write a digit",
            font=("Helvetica", 17, "bold"),
            fg=INK,
            bg=BACKGROUND,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self._build_drawing_column(root)
        self._build_result_column(root)

    def _build_drawing_column(self, root: tk.Frame) -> None:
        column = tk.Frame(root, bg=BACKGROUND)
        column.grid(row=1, column=0, sticky="n", padx=(0, 18))

        self.canvas = tk.Canvas(
            column,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground=TRACK,
            cursor="crosshair",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _event: self._schedule_prediction())
        self.bind("<KeyPress-c>", lambda _event: self._clear())
        self.focus_set()

        tk.Label(
            column,
            text="Draw one digit, large enough to fill most of the box.",
            font=("Helvetica", 11),
            fg=MUTED,
            bg=BACKGROUND,
        ).pack(pady=(10, 8))

        tk.Button(
            column,
            text="Clear  (c)",
            command=self._clear,
            font=("Helvetica", 12),
            highlightbackground=BACKGROUND,
        ).pack()

    def _build_result_column(self, root: tk.Frame) -> None:
        column = tk.Frame(root, bg=BACKGROUND)
        column.grid(row=1, column=1, sticky="n")

        self.digit_label = tk.Label(
            column, text="-", font=("Helvetica", 64, "bold"), fg=ACCENT, bg=BACKGROUND
        )
        self.digit_label.pack(anchor="w")

        self.confidence_label = tk.Label(
            column, text="", font=("Helvetica", 12), fg=MUTED, bg=BACKGROUND
        )
        self.confidence_label.pack(anchor="w", pady=(0, 10))

        self.bars = tk.Canvas(
            column,
            width=250,
            height=ROW_HEIGHT * 10,
            bg=PANEL,
            highlightthickness=0,
        )
        self.bars.pack(anchor="w")
        self._bar_items = [self._build_bar_row(digit) for digit in range(10)]

        tk.Label(
            column,
            text=f"model input ({IMAGE_SIZE}x{IMAGE_SIZE})",
            font=("Helvetica", 10),
            fg=MUTED,
            bg=BACKGROUND,
        ).pack(anchor="w", pady=(12, 4))

        self.preview = tk.Label(column, bg="black", bd=0)
        self.preview.pack(anchor="w")

    def _build_bar_row(self, digit: int) -> dict:
        """Create the label, track, fill and percentage of one probability row."""
        top = digit * ROW_HEIGHT
        middle = top + ROW_HEIGHT / 2
        self.bars.create_text(
            14, middle, text=str(digit), fill=MUTED, font=("Helvetica", 11, "bold")
        )
        self.bars.create_rectangle(
            BAR_LEFT, top + 6, BAR_RIGHT, top + ROW_HEIGHT - 6, fill=TRACK, width=0
        )
        return {
            "fill": self.bars.create_rectangle(
                BAR_LEFT, top + 6, BAR_LEFT, top + ROW_HEIGHT - 6, fill=ACCENT, width=0
            ),
            "percent": self.bars.create_text(
                BAR_RIGHT + 8,
                middle,
                text="0%",
                anchor="w",
                fill=MUTED,
                font=("Helvetica", 10),
            ),
            "top": top,
        }

    # --------------------------------------------------------------- drawing

    def _on_press(self, event: tk.Event) -> None:
        self._last_point = (event.x, event.y)
        radius = PEN_WIDTH / 2
        # A click without a drag should still leave a mark.
        self.canvas.create_oval(
            event.x - radius, event.y - radius, event.x + radius, event.y + radius,
            fill=INK, width=0,
        )
        self._stamp(event.x, event.y)
        self._schedule_prediction()

    def _on_drag(self, event: tk.Event) -> None:
        if self._last_point is None:
            self._on_press(event)
            return
        x0, y0 = self._last_point
        self.canvas.create_line(
            x0, y0, event.x, event.y,
            fill=INK, width=PEN_WIDTH, capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True,
        )
        self._draw.line([x0, y0, event.x, event.y], fill=255, width=PEN_WIDTH)
        # PIL draws butt caps, so round the ends off by hand to match the canvas.
        self._stamp(event.x, event.y)
        self._last_point = (event.x, event.y)
        self._schedule_prediction()

    def _stamp(self, x: int, y: int) -> None:
        radius = PEN_WIDTH / 2
        self._draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

    def _clear(self) -> None:
        self.canvas.delete("all")
        self._draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self._last_point = None
        self._show_blank()

    # ------------------------------------------------------------ prediction

    def _schedule_prediction(self) -> None:
        """Debounce: only classify once the pen has been still for a moment."""
        if self._pending_prediction is not None:
            self.after_cancel(self._pending_prediction)
        self._pending_prediction = self.after(PREDICT_DELAY_MS, self._predict)

    def _predict(self) -> None:
        self._pending_prediction = None
        digit_image = preprocess(self._image)
        if digit_image is None:
            self._show_blank()
            return
        probabilities = self.model.predict(
            digit_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1), verbose=0
        )[0]
        self._show_result(probabilities)
        self._show_preview(digit_image)

    def _show_result(self, probabilities: np.ndarray) -> None:
        best = int(np.argmax(probabilities))
        self.digit_label.config(text=str(best))
        self.confidence_label.config(text=f"{probabilities[best] * 100:.1f}% confident")
        for digit, items in enumerate(self._bar_items):
            probability = float(probabilities[digit])
            width = (BAR_RIGHT - BAR_LEFT) * probability
            self.bars.coords(
                items["fill"],
                BAR_LEFT,
                items["top"] + 6,
                BAR_LEFT + width,
                items["top"] + ROW_HEIGHT - 6,
            )
            self.bars.itemconfig(items["fill"], fill=ACCENT if digit == best else TRACK)
            self.bars.itemconfig(
                items["percent"],
                text=f"{probability * 100:.1f}%",
                fill=INK if digit == best else MUTED,
            )

    def _show_preview(self, digit_image: np.ndarray) -> None:
        """Show exactly what the network sees, magnified with hard pixel edges."""
        size = IMAGE_SIZE * PREVIEW_SCALE
        magnified = Image.fromarray((digit_image * 255.0).astype("uint8")).resize(
            (size, size), Image.NEAREST
        )
        self._preview_photo = ImageTk.PhotoImage(magnified)
        self.preview.config(image=self._preview_photo)

    def _show_blank(self) -> None:
        self.digit_label.config(text="-")
        self.confidence_label.config(text="nothing drawn yet")
        for items in self._bar_items:
            self.bars.coords(
                items["fill"], BAR_LEFT, items["top"] + 6, BAR_LEFT, items["top"] + ROW_HEIGHT - 6
            )
            self.bars.itemconfig(items["percent"], text="0.0%", fill=MUTED)
        self._show_preview(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype="float32"))


def main() -> None:
    DigitRecognizerApp().mainloop()


if __name__ == "__main__":
    main()
