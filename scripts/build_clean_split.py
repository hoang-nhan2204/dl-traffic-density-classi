from hashlib import sha256
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


SEED = 42

CLASS_TO_LABEL = {
    "Empty": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Traffic Jam": 4,
}

SPLITS = ["training", "validation", "testing"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "Final Dataset"
OUTPUT_DIR = PROJECT_DIR / "data" / "manifests"


def calculate_hash(image):
    image = image.convert("RGB")
    size = str(image.size).encode("utf-8")
    return sha256(size + image.tobytes()).hexdigest()


def scan_images():
    rows = []
    excluded = []

    for original_split in SPLITS:
        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = DATASET_DIR / original_split / class_name

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                relative_path = image_path.relative_to(PROJECT_DIR).as_posix()

                try:
                    with Image.open(image_path) as image:
                        image.load()
                        image_hash = calculate_hash(image)

                    rows.append(
                        {
                            "image_path": relative_path,
                            "label": label,
                            "class_name": class_name,
                            "original_split": original_split,
                            "image_hash": image_hash,
                        }
                    )

                except OSError:
                    excluded.append(
                        {
                            "image_path": relative_path,
                            "label": label,
                            "class_name": class_name,
                            "original_split": original_split,
                            "reason": "corrupted_image",
                            "representative_path": "",
                        }
                    )

    return pd.DataFrame(rows), excluded


def remove_duplicates(images_df):
    clean_rows = []
    excluded_rows = []

    for _, group in images_df.groupby("image_hash"):
        group = group.sort_values("image_path")

        if group["label"].nunique() > 1:
            for _, row in group.iterrows():
                excluded_rows.append(
                    {
                        "image_path": row["image_path"],
                        "label": row["label"],
                        "class_name": row["class_name"],
                        "original_split": row["original_split"],
                        "reason": "conflicting_label",
                        "representative_path": "",
                    }
                )

            continue

        representative = group.iloc[0]
        clean_rows.append(representative.to_dict())

        for _, row in group.iloc[1:].iterrows():
            excluded_rows.append(
                {
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "class_name": row["class_name"],
                    "original_split": row["original_split"],
                    "reason": "exact_duplicate",
                    "representative_path": representative["image_path"],
                }
            )

    return pd.DataFrame(clean_rows), excluded_rows


def create_splits(clean_df):
    train_df, remaining_df = train_test_split(
        clean_df,
        test_size=0.30,
        stratify=clean_df["label"],
        random_state=SEED,
    )

    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        stratify=remaining_df["label"],
        random_state=SEED,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def main():
    images_df, corrupted_images = scan_images()

    clean_df, duplicate_images = remove_duplicates(images_df)

    train_df, val_df, test_df = create_splits(clean_df)

    excluded_df = pd.DataFrame(
        corrupted_images + duplicate_images
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "image_path",
        "label",
        "class_name",
        "split",
        "original_split",
    ]

    train_df[columns].to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    val_df[columns].to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )

    test_df[columns].to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    excluded_df.to_csv(
        OUTPUT_DIR / "excluded_images.csv",
        index=False,
    )

    print(f"Total readable images: {len(images_df)}")
    print(f"Excluded images: {len(excluded_df)}")
    print(f"Train images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Test images: {len(test_df)}")

    print("\nExcluded by reason:")
    print(excluded_df["reason"].value_counts())

    print("\nClass distribution:")
    print(
        pd.crosstab(
            clean_df["class_name"],
            pd.concat([train_df, val_df, test_df])["split"],
        )
    )


if __name__ == "__main__":
    main()