"""Train the digit classifier on MNIST and save it for the GUI and the CLI.

Run from the project root:

    python3 -m digit_recognizer.train
    python3 -m digit_recognizer.train --epochs 20 --batch-size 256
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "digit_cnn.keras"


def use_certifi_ca_bundle() -> None:
    """Point Python at certifi's root certificates before MNIST is downloaded.

    The python.org builds for macOS ship without the system trust store wired
    up, so the download fails with CERTIFICATE_VERIFY_FAILED until an explicit
    bundle is configured.  Whatever the user already set always wins.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def load_mnist() -> tuple:
    """Fetch MNIST and shape it the way the model wants: (N, 28, 28, 1) in [0, 1]."""
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    return (x_train, y_train), (x_test, y_test)


def train(epochs: int, batch_size: int, output: Path) -> float:
    """Train, save the best weights to `output`, and return the test accuracy."""
    from tensorflow import keras

    from .model import build_model

    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"train: {x_train.shape[0]} images   test: {x_test.shape[0]} images")

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        # The checkpoint, not the final epoch, is what ends up on disk: with
        # augmentation on, the last epoch is not reliably the best one.
        keras.callbacks.ModelCheckpoint(
            str(output), monitor="val_accuracy", mode="max", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=2,
    )

    best = keras.models.load_model(str(output))
    _, accuracy = best.evaluate(x_test, y_test, verbose=0)
    print(f"\nsaved {output}")
    print(f"test accuracy: {accuracy:.4f}")
    return float(accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN to recognise handwritten digits.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    use_certifi_ca_bundle()
    train(args.epochs, args.batch_size, args.output)


if __name__ == "__main__":
    main()
