from pathlib import Path
import io
import os
import threading
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers


SEED = 42

CLASS_NAMES = [
    "Empty",
    "Low",
    "Medium",
    "High",
    "Traffic Jam",
]

PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_DIR / "data" / "manifests"
ARCHIVE_PATH = PROJECT_DIR / "traffic.zip"
_archive = None
_archive_lock = threading.Lock()


augmentation_layers = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(5 / 360, fill_mode="nearest", seed=SEED),
        layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=SEED),
        layers.RandomZoom(0.15, 0.15, fill_mode="nearest", seed=SEED),
        layers.RandomContrast(0.2, seed=SEED),
    ]
)


def read_image(image_path):
    if isinstance(image_path, bytes):
        image_path = image_path.decode("utf-8")
    else:
        image_path = str(image_path)

    source_path = Path(image_path)
    if source_path.exists():
        image_source = source_path
    else:
        # The project can be used without extracting Final Dataset/ locally.
        # A lock keeps shared ZipFile reads safe under tf.data parallel mapping.
        relative_path = source_path.relative_to(PROJECT_DIR).as_posix()
        global _archive
        with _archive_lock:
            if _archive is None:
                _archive = zipfile.ZipFile(ARCHIVE_PATH)
            image_source = io.BytesIO(_archive.read(relative_path))

    with Image.open(image_source) as image:
        image = image.convert("RGB")
        return np.array(image)


def load_image(image_path, label, image_size):
    image = tf.numpy_function(
        read_image,
        [image_path],
        tf.uint8,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize_with_pad(
        image,
        image_size[0],
        image_size[1],
    )

    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=len(CLASS_NAMES))

    return image, label


def augment_image(image, label):
    image = augmentation_layers(image, training=True)
    image = tf.image.random_brightness(
        image,
        max_delta=0.2,
        seed=SEED,
    )
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def make_dataset(
    dataframe,
    image_size,
    batch_size,
    training=False,
    augmentation=False,
    cache=True,
):
    image_paths = [
        str(PROJECT_DIR / image_path)
        for image_path in dataframe["image_path"]
    ]

    labels = dataframe["label"].to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    dataset = dataset.map(
        lambda path, label: load_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Cache decoded/resized immutable images in RAM. Augmentation deliberately
    # remains after cache so every epoch still receives new random transforms.
    if cache:
        dataset = dataset.cache()

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    if training and augmentation:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    options = tf.data.Options()
    options.threading.private_threadpool_size = min(16, os.cpu_count() or 1)
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_datasets(
    augmentation=True,
    image_size=(224, 224),
    batch_size=32,
):
    train_df = pd.read_csv(MANIFEST_DIR / "train.csv")
    val_df = pd.read_csv(MANIFEST_DIR / "val.csv")
    test_df = pd.read_csv(MANIFEST_DIR / "test.csv")

    train_dataset = make_dataset(
        train_df,
        image_size,
        batch_size,
        training=True,
        augmentation=augmentation,
    )

    val_dataset = make_dataset(
        val_df,
        image_size,
        batch_size,
    )

    test_dataset = make_dataset(
        test_df,
        image_size,
        batch_size,
    )

    return train_dataset, val_dataset, test_dataset
