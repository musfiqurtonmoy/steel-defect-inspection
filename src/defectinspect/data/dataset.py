"""PyTorch dataset backed by the SQL manifest."""

from __future__ import annotations

import random

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from defectinspect.data.preprocess import preprocess


class Augment:
    """Light, physically plausible augmentation for top-down surface images:
    flips, 90-degree rotations, and small brightness/contrast changes
    (lighting varies on a real line).

    Uses Python's `random`, which PyTorch reseeds in every DataLoader worker,
    so workers do not repeat each other's augmentations."""

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randrange(4)
        if k:
            img = img.rotate(90 * k)
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        return img


class ManifestDataset(Dataset):
    def __init__(self, items, classes, image_size=224, augment: Augment | None = None):
        self.items = items
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.augment is not None:
            img = self.augment(img)
        x = torch.from_numpy(preprocess(img, self.image_size))
        return x, self.class_to_idx[label]
