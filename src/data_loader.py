from pathlib import Path

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
    image_path = image_path.decode("utf-8")

    with Image.open(image_path) as image:
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
):
    image_paths = [
        str(PROJECT_DIR / image_path)
        for image_path in dataframe["image_path"]
    ]

    labels = dataframe["label"].to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
        )

    dataset = dataset.map(
        lambda path, label: load_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training and augmentation:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

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
