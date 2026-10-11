"""Transfer-learning data contract: final manifests, RGB letterbox and Keras batches."""

from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf


CLASS_NAMES = ["Empty", "Low", "Medium", "High", "Traffic Jam"]
PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_DIR / "data" / "transfer_final_manifests"


def letterbox(image, size=224):
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (124, 116, 103))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return np.asarray(canvas, dtype=np.float32)


class ManifestSequence(tf.keras.utils.Sequence):
    def __init__(self, split, batch_size=32, shuffle=False, seed=42):
        self.frame = pd.read_csv(MANIFEST_DIR / f"{split}.csv")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.archive = zipfile.ZipFile(PROJECT_DIR / "traffic.zip")
        self.indices = np.arange(len(self.frame))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.frame) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, index):
        ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        rows = self.frame.iloc[ids]
        images = []
        for path in rows["image_path"]:
            # Prefer extracted data when supplied by the team; otherwise read the
            # same manifest path directly from the immutable archive.
            source = PROJECT_DIR / path
            handle = source.open("rb") if source.exists() else io.BytesIO(self.archive.read(path))
            with handle, Image.open(handle) as image:
                images.append(letterbox(image))
        labels = tf.keras.utils.to_categorical(rows["label"].to_numpy(), num_classes=5)
        return np.stack(images), labels


def create_transfer_sequences(batch_size=32, seed=42):
    return (
        ManifestSequence("train", batch_size, shuffle=True, seed=seed),
        ManifestSequence("val", batch_size, shuffle=False, seed=seed),
        ManifestSequence("test", batch_size, shuffle=False, seed=seed),
    )
