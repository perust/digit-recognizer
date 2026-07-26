"""Classify handwritten digits stored in image files.

    python3 -m digit_recognizer.predict samples/*.png
    python3 -m digit_recognizer.predict scan.jpg --top 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .preprocess import ImageSource, as_model_input
from .train import DEFAULT_MODEL_PATH


def load_classifier(model_path: Path = DEFAULT_MODEL_PATH):
    """Load the trained model, with a pointed error when it has not been trained yet."""
    if not model_path.exists():
        raise SystemExit(
            f"no model at {model_path}\nTrain one first:  python3 -m digit_recognizer.train"
        )
    from tensorflow import keras

    return keras.models.load_model(str(model_path))


def predict_probabilities(model, image: ImageSource) -> Optional[np.ndarray]:
    """Return the 10 class probabilities, or None if the image holds no ink."""
    batch = as_model_input(image)
    if batch is None:
        return None
    return model.predict(batch, verbose=0)[0]


def format_prediction(probabilities: np.ndarray, top: int = 3) -> str:
    """Render the `top` most likely digits as 'digit 92.1%' pairs."""
    ranked = np.argsort(probabilities)[::-1][:top]
    return "  ".join(f"{digit} {probabilities[digit] * 100:5.1f}%" for digit in ranked)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognise handwritten digits in images.")
    parser.add_argument("images", nargs="+", type=Path, help="image files, one digit each")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top", type=int, default=3, help="how many candidates to show")
    args = parser.parse_args()

    model = load_classifier(args.model)
    failures = 0
    for path in args.images:
        try:
            probabilities = predict_probabilities(model, path)
        except OSError as error:  # unreadable file, or not an image at all
            print(f"{path.name:<28} error: {error}")
            failures += 1
            continue
        if probabilities is None:
            print(f"{path.name:<28} blank (no strokes found)")
            continue
        digit = int(np.argmax(probabilities))
        print(f"{path.name:<28} -> {digit}    {format_prediction(probabilities, args.top)}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
