"""Checks on the normalisation contract the model depends on.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digit_recognizer.preprocess import DIGIT_BOX, IMAGE_SIZE, preprocess  # noqa: E402


def draw_seven(size: int = 300, scale: float = 1.0, offset: tuple[int, int] = (0, 0)) -> Image.Image:
    """Rasterise a "7"-like stroke: white on black, like the drawing canvas."""
    image = Image.new("L", (size, size), color=0)
    pen = ImageDraw.Draw(image)
    centre = size / 2
    stroke = [(-60, -70), (60, -70), (-10, 80)]
    points = [(centre + x * scale + offset[0], centre + y * scale + offset[1]) for x, y in stroke]
    pen.line(points, fill=255, width=max(2, int(18 * scale)), joint="curve")
    return image


def ink_bounds(digit: np.ndarray, threshold: float = 0.12) -> tuple[int, int]:
    """Height and width of the tight box around the strokes."""
    mask = digit > threshold
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1


def center_of_mass(digit: np.ndarray) -> tuple[float, float]:
    rows, cols = np.mgrid[0 : digit.shape[0], 0 : digit.shape[1]]
    total = digit.sum()
    return float((digit * rows).sum() / total), float((digit * cols).sum() / total)


class PreprocessOutputTest(unittest.TestCase):
    def test_shape_and_range(self):
        digit = preprocess(draw_seven())
        self.assertEqual(digit.shape, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(digit.dtype, np.float32)
        self.assertGreaterEqual(digit.min(), 0.0)
        self.assertLessEqual(digit.max(), 1.0)

    def test_blank_image_returns_none(self):
        self.assertIsNone(preprocess(Image.new("L", (300, 300), color=0)))
        self.assertIsNone(preprocess(Image.new("L", (300, 300), color=255)))


class MnistConventionTest(unittest.TestCase):
    """The two rules the original dataset applied: 20x20 box, centre of mass centred."""

    def test_digit_fits_the_20x20_box(self):
        for scale in (0.4, 1.0, 1.6):
            with self.subTest(scale=scale):
                height, width = ink_bounds(preprocess(draw_seven(scale=scale)))
                self.assertLessEqual(max(height, width), DIGIT_BOX)
                self.assertGreaterEqual(max(height, width), DIGIT_BOX - 2)

    def test_centre_of_mass_lands_in_the_middle(self):
        for offset in ((0, 0), (-70, 60), (55, -45)):
            with self.subTest(offset=offset):
                row, col = center_of_mass(preprocess(draw_seven(offset=offset)))
                middle = (IMAGE_SIZE - 1) / 2.0
                self.assertAlmostEqual(row, middle, delta=1.0)
                self.assertAlmostEqual(col, middle, delta=1.0)

    def test_translation_is_removed_exactly(self):
        centred = preprocess(draw_seven())
        shifted = preprocess(draw_seven(offset=(40, -30)))
        np.testing.assert_allclose(centred, shifted, atol=1e-6)


class InputFormatTest(unittest.TestCase):
    """Anything that reads as "one digit on a background" should work."""

    def test_dark_ink_on_paper_is_inverted(self):
        drawn = draw_seven()
        scanned = Image.fromarray(255 - np.asarray(drawn))
        np.testing.assert_allclose(preprocess(drawn), preprocess(scanned), atol=1e-6)

    def test_low_contrast_pencil_is_stretched(self):
        faint = Image.fromarray((np.asarray(draw_seven(), dtype=np.float32) * 0.35).astype("uint8"))
        np.testing.assert_allclose(preprocess(draw_seven()), preprocess(faint), atol=0.05)

    def test_transparent_background_is_not_mistaken_for_ink(self):
        ink = np.asarray(draw_seven())
        # Black strokes on a fully transparent background, as a drawing app exports.
        rgba = np.zeros((*ink.shape, 4), dtype="uint8")
        rgba[..., 3] = ink
        np.testing.assert_allclose(
            preprocess(draw_seven()), preprocess(Image.fromarray(rgba, "RGBA")), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
