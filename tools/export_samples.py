"""Write MNIST test digits out as ordinary image files.

The point is to exercise the same path a real photo would take: the files come
out large, dark-ink-on-white and slightly blurred, so running the CLI over them
checks inversion, level normalisation, cropping and centring -- not just the
network.  The true label is baked into each filename.

    python3 tools/export_samples.py --count 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digit_recognizer.train import use_certifi_ca_bundle  # noqa: E402

OUTPUT_SIZE = 280


def export(count: int, output_dir: Path, offset: int) -> None:
    use_certifi_ca_bundle()
    from tensorflow import keras

    (_, _), (images, labels) = keras.datasets.mnist.load_data()
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(offset, offset + count):
        page = Image.fromarray(255 - images[index])  # black ink on white paper
        page = page.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BICUBIC)
        page = page.filter(ImageFilter.GaussianBlur(1.2))  # soften the upscaled edges
        page.save(output_dir / f"digit_{index:05d}_label{labels[index]}.png")

    print(f"wrote {count} images to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MNIST test digits as PNG files.")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "samples"
    )
    args = parser.parse_args()
    export(args.count, args.output_dir, args.offset)


if __name__ == "__main__":
    main()
