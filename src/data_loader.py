from pathlib import Path

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def create_generators(
    augmentation=True,
    image_size=(224, 224),
    batch_size=32,
):
    train_df = pd.read_csv(MANIFEST_DIR / "train.csv")
    val_df = pd.read_csv(MANIFEST_DIR / "val.csv")
    test_df = pd.read_csv(MANIFEST_DIR / "test.csv")

    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=5,
            width_shift_range=0.08,
            height_shift_range=0.08,
            shear_range=5,
            zoom_range=(0.85, 1.15),
            brightness_range=(0.6, 1.4),
            horizontal_flip=True,
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1 / 255,
        )

    plain_datagen = ImageDataGenerator(
        rescale=1 / 255,
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=str(PROJECT_DIR),
        x_col="image_path",
        y_col="class_name",
        classes=CLASS_NAMES,
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
    )

    val_generator = plain_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=str(PROJECT_DIR),
        x_col="image_path",
        y_col="class_name",
        classes=CLASS_NAMES,
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )

    test_generator = plain_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=str(PROJECT_DIR),
        x_col="image_path",
        y_col="class_name",
        classes=CLASS_NAMES,
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator