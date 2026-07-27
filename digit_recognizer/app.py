"""A drawing board that recognises the digit you write on it.

    python3 -m digit_recognizer.app

Draw with the mouse; the prediction refreshes shortly after you stop moving.
Rest the pen and the digit adds itself to the field along the bottom, so a
longer number can be written a digit at a time and then copied out. Enter adds
it immediately, Backspace takes the last one back, and "c" clears the pad.
"""

from __future__ import annotations

import time
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

# How long the pen has to rest before the digit is added on its own.  Kept well
# above a second because 4, 5 and a crossed 7 take two strokes, and the gap
# while the mouse travels to the second one is easily most of a second; commit
# too eagerly and a "4" is filed as "1" followed by another "1".
AUTO_COMMIT_MS = 1200
COUNTDOWN_TICK_MS = 25  # how often the progress bar is redrawn
COUNTDOWN_HEIGHT = 4

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

        # What the board currently reads, and the digits committed so far.
        self._reading: int | None = None
        self.digits = tk.StringVar()

        self._drawing = False
        self.auto_commit = tk.BooleanVar(value=True)
        self._auto_deadline: float | None = None
        self._countdown_job: str | None = None

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
        self._build_collector(root)

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
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<KeyPress-c>", self._on_clear_key)
        self.bind("<Return>", lambda _event: self._commit_digit())
        self.bind("<BackSpace>", self._on_backspace)
        self.focus_set()

        # Fills up while the pen rests, then the digit is added. Showing the
        # wait rather than hiding it is what makes the automatic add legible:
        # you can see it coming and carry on drawing to call it off.
        self.countdown = tk.Canvas(
            column, width=CANVAS_SIZE, height=COUNTDOWN_HEIGHT, bg=BACKGROUND, highlightthickness=0
        )
        self.countdown.pack(pady=(4, 0))
        self._countdown_bar = self.countdown.create_rectangle(
            0, 0, 0, COUNTDOWN_HEIGHT, fill=ACCENT, width=0
        )

        tk.Label(
            column,
            text="Draw one digit large, then pause — or press Enter.",
            font=("Helvetica", 11),
            fg=MUTED,
            bg=BACKGROUND,
        ).pack(pady=(8, 8))

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

    def _build_collector(self, root: tk.Frame) -> None:
        """The strip along the bottom where recognised digits pile up.

        An ordinary entry rather than a read-only label, so the digits can be
        corrected by hand, typed into directly, and selected for copying like
        text from anywhere else.
        """
        row = tk.Frame(root, bg=BACKGROUND)
        row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(16, 0))

        tk.Checkbutton(
            root,
            text=f"Add on its own after a {AUTO_COMMIT_MS / 1000:.1f}s pause",
            variable=self.auto_commit,
            command=self._on_auto_commit_toggled,
            font=("Helvetica", 11),
            fg=MUTED,
            bg=BACKGROUND,
            activebackground=BACKGROUND,
            activeforeground=INK,
            selectcolor=PANEL,
            highlightthickness=0,
            bd=0,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        tk.Label(
            row, text="Digits", font=("Helvetica", 11), fg=MUTED, bg=BACKGROUND
        ).pack(side="left", padx=(0, 8))

        self.entry = tk.Entry(
            row,
            textvariable=self.digits,
            font=("Helvetica", 16),
            fg=INK,
            bg=PANEL,
            insertbackground=INK,
            relief="flat",
            highlightthickness=1,
            highlightbackground=TRACK,
            highlightcolor=ACCENT,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=5)

        self.copy_button = tk.Button(
            row, text="Copy", command=self._copy_digits,
            font=("Helvetica", 12), highlightbackground=BACKGROUND, width=6,
        )
        for button in (
            tk.Button(
                row, text="Add  ⏎", command=self._commit_digit,
                font=("Helvetica", 12), highlightbackground=BACKGROUND,
            ),
            tk.Button(
                row, text="Undo  ⌫", command=self._undo_digit,
                font=("Helvetica", 12), highlightbackground=BACKGROUND,
            ),
            self.copy_button,
            tk.Button(
                row, text="Erase", command=lambda: self.digits.set(""),
                font=("Helvetica", 12), highlightbackground=BACKGROUND,
            ),
        ):
            button.pack(side="left", padx=(8, 0))

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
        self._drawing = True
        self._cancel_auto_commit()  # a new stroke means the digit is not finished
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

    def _on_release(self, _event: tk.Event) -> None:
        self._drawing = False
        self._schedule_prediction()

    def _stamp(self, x: int, y: int) -> None:
        radius = PEN_WIDTH / 2
        self._draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

    def _clear(self) -> None:
        self._cancel_auto_commit()
        self.canvas.delete("all")
        self._draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self._last_point = None
        self._show_blank()

    def _on_clear_key(self, _event: tk.Event) -> None:
        """"c" wipes the pad -- unless it is being typed into the digits field."""
        if self.focus_get() is self.entry:
            return
        self._clear()

    # -------------------------------------------------------- automatic adding

    def _start_auto_commit(self) -> None:
        """Begin the wait after which a resting digit adds itself."""
        self._auto_deadline = time.monotonic() + AUTO_COMMIT_MS / 1000
        self._tick_countdown()

    def _cancel_auto_commit(self) -> None:
        self._auto_deadline = None
        if self._countdown_job is not None:
            self.after_cancel(self._countdown_job)
            self._countdown_job = None
        self.countdown.coords(self._countdown_bar, 0, 0, 0, COUNTDOWN_HEIGHT)

    def _tick_countdown(self) -> None:
        """Advance the bar, and commit once it is full."""
        self._countdown_job = None
        if self._auto_deadline is None:
            return

        remaining = self._auto_deadline - time.monotonic()
        if remaining <= 0:
            self._auto_deadline = None
            self._commit_digit()
            return

        elapsed = 1 - remaining / (AUTO_COMMIT_MS / 1000)
        self.countdown.coords(
            self._countdown_bar, 0, 0, CANVAS_SIZE * elapsed, COUNTDOWN_HEIGHT
        )
        self._countdown_job = self.after(COUNTDOWN_TICK_MS, self._tick_countdown)

    def _on_auto_commit_toggled(self) -> None:
        if self.auto_commit.get():
            if self._reading is not None and not self._drawing:
                self._start_auto_commit()
        else:
            self._cancel_auto_commit()

    # ----------------------------------------------------------- collected text

    def _commit_digit(self) -> None:
        """Append what the board currently reads and make room for the next digit."""
        if self._reading is None:
            return
        self.digits.set(self.digits.get() + str(self._reading))
        self.entry.icursor(tk.END)
        self._clear()
        # Back to the window itself, so the next "c" reaches the pad rather
        # than typing a letter into the field the Add button may have focused.
        self.focus_set()

    def _undo_digit(self) -> None:
        """Drop the digit added last.

        The safety net for the automatic add: when a two-stroke digit gets
        filed halfway through, one press takes it back.
        """
        self.digits.set(self.digits.get()[:-1])
        self.entry.icursor(tk.END)

    def _on_backspace(self, _event: tk.Event) -> None:
        # Inside the field, Backspace is ordinary text editing; outside it,
        # it undoes the last digit that was collected.
        if self.focus_get() is self.entry:
            return
        self._undo_digit()

    def _copy_digits(self) -> None:
        text = self.digits.get()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        # Tk hands the selection over lazily; without this the clipboard is
        # empty for anyone who pastes after the window has gone away.
        self.update()
        self._flash(self.copy_button, "Copied")

    def _flash(self, button: tk.Button, message: str, milliseconds: int = 1200) -> None:
        """Say something happened on the button itself, then put the label back."""
        original = button.cget("text")
        button.config(text=message)
        button.after(milliseconds, lambda: button.config(text=original))

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
        # Only once the pen is up: holding still mid-stroke is not a finished digit.
        if self.auto_commit.get() and not self._drawing:
            self._start_auto_commit()

    def _show_result(self, probabilities: np.ndarray) -> None:
        best = int(np.argmax(probabilities))
        self._reading = best
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
        self._reading = None
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
