"""Export the trained network in a form a browser can run without a library.

    python3 tools/export_web_model.py

Writes web/model.json (the layer plan and the quantisation scales) and
web/weights.bin (the numbers).  Two transformations happen on the way out:

*   Each batch-norm is folded into the convolution or dense layer in front of
    it.  At inference a batch-norm is only an affine map per channel, so it can
    be multiplied into the weights it follows, leaving the browser with nothing
    but conv, dense, pool and relu to implement.
*   Weights are quantised to int8 with one scale per output channel, which is
    what keeps the download under a megabyte.  Per-channel rather than
    per-tensor matters here: the dense layer's columns differ in magnitude by
    more than an order of magnitude, and a single shared scale would round the
    small ones away.

The dropout and augmentation layers are inference-time no-ops and are dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_MODEL = PROJECT_DIR / "models" / "digit_cnn.keras"
DEFAULT_OUTPUT = PROJECT_DIR / "web"

SKIPPED = ("RandomRotation", "RandomZoom", "RandomTranslation", "Dropout")


class Writer:
    """Collects tensors into one binary blob and remembers where each landed."""

    def __init__(self) -> None:
        self.blob = bytearray()
        self.tensors: list[dict] = []

    def add(self, array: np.ndarray, dtype: str) -> int:
        # Float views in JavaScript must start on a 4-byte boundary, so pad
        # rather than let a preceding int8 tensor push them out of alignment.
        while len(self.blob) % 4:
            self.blob.append(0)
        entry = {"dtype": dtype, "offset": len(self.blob), "count": int(array.size)}
        self.blob.extend(array.astype({"int8": np.int8, "float32": np.float32}[dtype]).tobytes())
        self.tensors.append(entry)
        return len(self.tensors) - 1

    def add_quantised(self, weights: np.ndarray) -> int:
        """Store `weights` as int8, scaled per output channel (the last axis)."""
        peaks = np.abs(weights).reshape(-1, weights.shape[-1]).max(axis=0)
        scales = np.where(peaks > 0, peaks / 127.0, 1.0).astype(np.float32)
        quantised = np.clip(np.rint(weights / scales), -127, 127).astype(np.int8)
        index = self.add(quantised, "int8")
        self.tensors[index]["scales"] = [float(s) for s in scales]
        return index


def fold_batch_norm(weights: np.ndarray, norm) -> tuple[np.ndarray, np.ndarray]:
    """Multiply a batch-norm into the layer it follows.

    Inference-time batch-norm is `gamma * (x - mean) / sqrt(var + eps) + beta`
    applied per output channel.  Because the layer in front carries no bias of
    its own, the whole thing collapses into scaled weights plus a new bias.
    """
    gamma, beta, mean, variance = (np.asarray(w) for w in norm.get_weights())
    scale = gamma / np.sqrt(variance + norm.epsilon)
    return weights * scale, beta - mean * scale


def build_plan(model, writer: Writer) -> list[dict]:
    """Walk the Keras layers and emit the equivalent list of browser ops."""
    layers = [layer for layer in model.layers if type(layer).__name__ not in SKIPPED]
    plan: list[dict] = []
    index = 0

    while index < len(layers):
        layer = layers[index]
        kind = type(layer).__name__

        if kind in ("Conv2D", "Dense"):
            weights = np.asarray(layer.get_weights()[0])
            following = layers[index + 1] if index + 1 < len(layers) else None
            if following is not None and type(following).__name__ == "BatchNormalization":
                weights, bias = fold_batch_norm(weights, following)
                index += 1  # the batch-norm is now part of this layer
            else:
                bias = np.asarray(layer.get_weights()[1])

            activation = layer.get_config().get("activation", "linear")
            following = layers[index + 1] if index + 1 < len(layers) else None
            if following is not None and type(following).__name__ == "Activation":
                activation = following.get_config()["activation"]
                index += 1

            plan.append(
                {
                    "type": "conv" if kind == "Conv2D" else "dense",
                    "shape": list(weights.shape),
                    "activation": activation,
                    "weights": writer.add_quantised(weights),
                    "bias": writer.add(bias, "float32"),
                }
            )
        elif kind == "MaxPooling2D":
            plan.append({"type": "maxpool", "size": int(layer.pool_size[0])})
        elif kind == "Flatten":
            plan.append({"type": "flatten"})
        else:
            raise SystemExit(f"export_web_model: no browser equivalent for {kind}")
        index += 1

    return plan


def run_plan(plan: list[dict], tensors: list[dict], blob: bytes, image: np.ndarray) -> np.ndarray:
    """Reference implementation of the exported plan, used to check the export.

    Mirrors what web/digit-model.js does, so a disagreement between this and
    the original Keras model localises the bug to the export rather than to the
    JavaScript.
    """
    def read(index: int, shape: tuple | None = None) -> np.ndarray:
        entry = tensors[index]
        dtype = np.int8 if entry["dtype"] == "int8" else np.float32
        raw = np.frombuffer(blob, dtype=dtype, count=entry["count"], offset=entry["offset"])
        values = raw.astype(np.float32)
        if shape is not None:
            values = values.reshape(shape)
        # The scales are per output channel, which is the last axis of the
        # reshaped tensor, so the multiply has to happen after the reshape.
        return values * np.asarray(entry["scales"], np.float32) if "scales" in entry else values

    activations = image.reshape(28, 28, 1).astype(np.float32)
    for step in plan:
        if step["type"] == "conv":
            kernel = read(step["weights"], tuple(step["shape"]))
            padded = np.pad(activations, ((1, 1), (1, 1), (0, 0)))
            height, width = activations.shape[:2]
            patches = np.stack(
                [padded[y : y + height, x : x + width] for y in range(3) for x in range(3)]
            )  # (9, H, W, in)
            activations = np.einsum("nhwi,nio->hwo", patches, kernel.reshape(9, *kernel.shape[2:]))
            activations += read(step["bias"])
        elif step["type"] == "maxpool":
            size = step["size"]
            height, width, channels = activations.shape
            activations = activations.reshape(
                height // size, size, width // size, size, channels
            ).max(axis=(1, 3))
        elif step["type"] == "flatten":
            activations = activations.reshape(-1)
        elif step["type"] == "dense":
            activations = activations @ read(step["weights"], tuple(step["shape"]))
            activations += read(step["bias"])

        if step.get("activation") == "relu":
            activations = np.maximum(activations, 0.0)
        elif step.get("activation") == "softmax":
            shifted = np.exp(activations - activations.max())
            activations = shifted / shifted.sum()

    return activations


def export(model_path: Path, output_dir: Path) -> None:
    from tensorflow import keras

    from digit_recognizer.train import use_certifi_ca_bundle

    if not model_path.exists():
        raise SystemExit(f"no model at {model_path}\nTrain one first: python3 -m digit_recognizer.train")

    model = keras.models.load_model(str(model_path))
    writer = Writer()
    plan = build_plan(model, writer)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weights.bin").write_bytes(bytes(writer.blob))
    (output_dir / "model.json").write_text(
        json.dumps(
            {
                "input": [28, 28, 1],
                "classes": 10,
                "layers": plan,
                "tensors": writer.tensors,
                "bytes": len(writer.blob),
            },
            indent=1,
        )
    )

    for step in plan:
        detail = f" {step['shape']}" if "shape" in step else ""
        print(f"  {step['type']}{detail} {step.get('activation', '')}".rstrip())
    print(f"\nweights.bin  {len(writer.blob) / 1024:.0f} KB")

    _report_accuracy(model, plan, writer, use_certifi_ca_bundle)


def _report_accuracy(model, plan, writer, use_certifi_ca_bundle) -> None:
    """Measure what the quantisation actually cost, on the MNIST test set."""
    from tensorflow import keras

    use_certifi_ca_bundle()
    (_, _), (images, labels) = keras.datasets.mnist.load_data()
    images = images.astype("float32") / 255.0

    reference = model.predict(images[..., np.newaxis], verbose=0).argmax(axis=1)
    print(f"float model : {(reference == labels).mean():.4f}")

    blob = bytes(writer.blob)
    exported = np.array([run_plan(plan, writer.tensors, blob, image).argmax() for image in images])
    print(f"int8 export : {(exported == labels).mean():.4f}")
    print(f"agreement   : {(exported == reference).mean():.4f} of predictions identical")

    _write_parity_fixture(plan, writer, blob, images, labels)


def _write_parity_fixture(plan, writer, blob, images, labels, count: int = 20) -> None:
    """Record inputs and outputs so the JavaScript port can be held to them.

    Without this the browser code could drift -- a transposed kernel, a missed
    dequantisation scale -- and still look plausible, because a digit drawn by
    hand has no ground truth to compare against.  These do.
    """
    import base64

    sample = images[:count]
    probabilities = [run_plan(plan, writer.tensors, blob, image) for image in sample]
    fixture = PROJECT_DIR / "tests" / "fixtures" / "model_parity.json"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(
        json.dumps(
            {
                "note": "inputs are 28x28 uint8 pixels, base64; probabilities come from run_plan in tools/export_web_model.py",
                "labels": [int(label) for label in labels[:count]],
                "inputs": base64.b64encode(np.rint(sample * 255).astype(np.uint8).tobytes()).decode(),
                "probabilities": [[float(p) for p in row] for row in probabilities],
            },
            indent=1,
        )
    )
    print(f"fixture     : {fixture.relative_to(PROJECT_DIR)} ({count} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the model for the browser.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.model, args.output_dir)


if __name__ == "__main__":
    main()
