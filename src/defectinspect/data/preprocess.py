"""Preprocessing shared by training, evaluation and the API.

Kept in NumPy and PIL so the serving container does not need PyTorch.
Training and serving call the same function, so there is no train/serve skew.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_rgb(img: Image.Image) -> Image.Image:
    """Grayscale steel images are copied to three channels for ImageNet backbones."""
    return img.convert("RGB")


def preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """PIL image -> float32 array of shape (3, size, size), normalised."""
    img = to_rgb(img).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1).copy()


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
