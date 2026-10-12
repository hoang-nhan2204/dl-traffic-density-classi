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
SHARED_AUGMENTATION_NAME = "shared_cnn_v1"


def letterbox(image, size=224):
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (124, 116, 103))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return np.asarray(canvas, dtype=np.float32)


@tf.keras.utils.register_keras_serializable(package="traffic_density")
class SharedCnnAugmentation(tf.keras.layers.Layer):
    """Apply the exact train-time augmentation policy used by the CNN notebooks.

    ``ManifestSequence`` returns raw RGB values in [0, 255] because the
    pretrained Keras applications expect that scale.  The shared CNN pipeline
    applies its transforms to [0, 1] images, so this layer temporarily rescales
    only while augmenting and restores [0, 255] before a backbone-specific
    ImageNet preprocessor is called.
    """

    def __init__(self, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.transforms = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", seed=seed),
                tf.keras.layers.RandomRotation(
                    5 / 360,
                    fill_mode="nearest",
                    seed=seed,
                ),
                tf.keras.layers.RandomTranslation(
                    0.08,
                    0.08,
                    fill_mode="nearest",
                    seed=seed,
                ),
                tf.keras.layers.RandomZoom(
                    0.15,
                    0.15,
                    fill_mode="nearest",
                    seed=seed,
                ),
                tf.keras.layers.RandomContrast(0.2, seed=seed),
            ],
            name="shared_cnn_geometric_and_contrast",
        )

    def _augment(self, images):
        images = tf.cast(images, tf.float32) / 255.0
        images = self.transforms(images, training=True)
        images = tf.image.random_brightness(
            images,
            max_delta=0.2,
            seed=self.seed,
        )
        images = tf.clip_by_value(images, 0.0, 1.0)
        return images * 255.0

    def build(self, input_shape):
        self.transforms.build(input_shape)
        super().build(input_shape)

    def call(self, images, training=None):
        if training is True:
            return self._augment(images)
        if training is False or training is None:
            return images
        return tf.cond(training, lambda: self._augment(images), lambda: images)

    def get_config(self):
        return {**super().get_config(), "seed": self.seed}


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


def as_tensor_dataset(sequence):
    """Expose an existing deterministic sequence as tf.data for shared evaluation.

    The shared ``evaluate_model`` helper expects Tensor labels and can therefore
    be used without altering the Simple/Complex CNN evaluation module.
    """

    images, labels = sequence[0]
    signature = (
        tf.TensorSpec(shape=(None, *images.shape[1:]), dtype=tf.float32),
        tf.TensorSpec(shape=(None, labels.shape[1]), dtype=tf.float32),
    )

    def batches():
        for index in range(len(sequence)):
            batch_images, batch_labels = sequence[index]
            yield batch_images.astype(np.float32), batch_labels.astype(np.float32)

    return tf.data.Dataset.from_generator(batches, output_signature=signature)
