"""The convolutional network that classifies 28x28 handwritten digits."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from .preprocess import IMAGE_SIZE

NUM_CLASSES = 10


def build_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """A small VGG-style CNN. Input pixels are expected to be scaled to [0, 1].

    The random rotation / zoom / shift layers at the front only perturb their
    input while `fit` is running; `predict` runs them in inference mode, where
    they pass the image straight through.  Keeping them inside the model means
    the saved file carries its own training recipe, and the augmentation is what
    lets a network trained on neat scanned digits cope with the wobblier strokes
    that come out of a mouse-drawn canvas.
    """
    return keras.Sequential(
        [
            keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="digit"),
            # Background is black, so vacated pixels must be filled with 0 --
            # the default "reflect" would smear copies of the stroke inwards.
            layers.RandomRotation(0.06, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
            *_conv_block(32),
            *_conv_block(64),
            layers.Flatten(),
            layers.Dense(256, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax", name="probabilities"),
        ],
        name="digit_cnn",
    )


def _conv_block(filters: int) -> list:
    """Two 3x3 convolutions, then halve the resolution.

    Bias terms are dropped because the batch-norm that follows immediately
    re-centres the activations, making them redundant parameters.
    """
    return [
        layers.Conv2D(filters, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(filters, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
    ]
