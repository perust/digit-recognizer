"""Normalise a picture of one handwritten digit into MNIST's exact format.

MNIST images are not plain 28x28 crops.  Every digit in the original dataset
was scaled to fit inside a 20x20 box with its aspect ratio preserved, and then
translated so that its centre of mass landed on the centre of a 28x28 frame.
A network trained on that distribution only behaves well when the pictures it
is asked to classify went through the very same normalisation, which is why the
GUI and the command line tool both funnel their input through this module.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
from PIL import Image

IMAGE_SIZE = 28  # side length of the frame the model expects
DIGIT_BOX = 20  # side length of the box the digit is scaled to fit
INK_THRESHOLD = 0.12  # pixels above this (on a 0..1 scale) count as ink

ImageSource = Union[Image.Image, str, "os.PathLike[str]"]


def to_grayscale(image: ImageSource) -> np.ndarray:
    """Load `image` if needed and return it as a float32 array in [0, 1]."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return np.asarray(_drop_alpha(image), dtype=np.float32) / 255.0


def _drop_alpha(image: Image.Image) -> Image.Image:
    """Flatten any transparency onto white before going grayscale.

    Converting an RGBA image straight to "L" throws the alpha channel away and
    turns transparent pixels black, which would look like ink to everything
    downstream.  Compositing onto white keeps them as background instead.
    """
    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba)
    return image.convert("L")


def ensure_white_on_black(gray: np.ndarray) -> np.ndarray:
    """Invert the image when it holds dark ink on a light page.

    MNIST stores white strokes on a black background, but a photo or a scan of
    paper is the other way round.  The border of the picture is almost always
    background, so a bright border means the image needs flipping.
    """
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return 1.0 - gray if float(np.median(border)) > 0.5 else gray


def normalise_levels(gray: np.ndarray) -> np.ndarray:
    """Push the paper to pure black and the strongest stroke to pure white.

    Photographs leave a grey haze once inverted, and pencil never reaches full
    intensity.  Subtracting the level measured on the border and then stretching
    what is left brings both closer to the crisp statistics of the training set.
    """
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    cleaned = np.clip(gray - float(np.median(border)), 0.0, 1.0)
    peak = float(cleaned.max())
    return cleaned / peak if peak > 0.0 else cleaned


def crop_to_ink(
    gray: np.ndarray, threshold: float = INK_THRESHOLD
) -> Optional[np.ndarray]:
    """Crop to the tight bounding box of the strokes, or None if nothing is drawn."""
    mask = gray > threshold
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def fit_in_box(digit: np.ndarray, box: int = DIGIT_BOX) -> np.ndarray:
    """Scale the cropped digit so its longer side is `box` pixels."""
    height, width = digit.shape
    scale = box / max(height, width)
    # Clamp to one pixel so a thin, tall "1" cannot collapse to zero width.
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray((digit * 255.0).astype(np.uint8)).resize(
        target, Image.LANCZOS
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def center_by_mass(digit: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Lay the digit on a `size` x `size` canvas, centre of mass in the middle.

    Centring by mass rather than by bounding box is what the original dataset
    did, and it matters: the ink of a "7" sits high in its own bounding box, so
    the two conventions place it several pixels apart.
    """
    canvas = np.zeros((size, size), dtype=np.float32)
    height, width = digit.shape
    top, left = (size - height) // 2, (size - width) // 2
    canvas[top : top + height, left : left + width] = digit

    total = float(canvas.sum())
    if total == 0.0:
        return canvas
    rows, cols = np.mgrid[0:size, 0:size]
    middle = (size - 1) / 2.0
    dx = round(middle - float((canvas * cols).sum()) / total)
    dy = round(middle - float((canvas * rows).sum()) / total)
    return _translate(canvas, dx, dy)


def _translate(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift by whole pixels, filling the vacated border with background."""
    height, width = array.shape
    # Clamping keeps the slice arithmetic below in range; a shift of a full side
    # would otherwise wrap into a negative stop and copy the wrong rows.
    dy = max(-height, min(height, dy))
    dx = max(-width, min(width, dx))

    shifted = np.zeros_like(array)
    source = (slice(max(0, -dy), height - max(0, dy)), slice(max(0, -dx), width - max(0, dx)))
    target = (slice(max(0, dy), height - max(0, -dy)), slice(max(0, dx), width - max(0, -dx)))
    shifted[target] = array[source]
    return shifted


def preprocess(image: ImageSource) -> Optional[np.ndarray]:
    """Run the full pipeline: any picture of one digit -> (28, 28) float32 in [0, 1].

    Returns None when the image holds no strokes at all, so that callers can
    tell an empty canvas apart from a prediction the model is unsure about.
    """
    gray = normalise_levels(ensure_white_on_black(to_grayscale(image)))
    digit = crop_to_ink(gray)
    if digit is None:
        return None
    return center_by_mass(fit_in_box(digit))


def as_model_input(image: ImageSource) -> Optional[np.ndarray]:
    """Preprocess and add the batch and channel axes the model expects."""
    digit = preprocess(image)
    if digit is None:
        return None
    return digit.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
