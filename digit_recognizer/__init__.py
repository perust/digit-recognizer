"""Handwritten digit recognition: MNIST-style preprocessing plus a small CNN.

The package is split so that the training script, the command line classifier
and the drawing GUI all share one definition of what an input image should look
like by the time it reaches the network.  See `preprocess.py` for why that
matters more than the architecture does.
"""

__all__ = ["preprocess", "model"]
