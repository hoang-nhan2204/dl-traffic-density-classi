"""Generate Grad-CAM explanations for the frozen final transfer checkpoint.

This is post-test diagnostic analysis only.  It reads the immutable saved test
manifest and checkpoint; it neither trains nor changes model selection.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.transfer_data import CLASS_NAMES, PROJECT_DIR, create_transfer_sequences


DEFAULT_CHECKPOINT = (
    PROJECT_DIR
    / "outputs"
    / "transfer_learning_shared_aug_v2"
    / "efficientnetb0_A0_F2_ce_s42"
    / "best.keras"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "outputs"
    / "gradcam"
    / "efficientnetb0_A0_F2_ce_s42"
)


def find_last_conv2d(layer: tf.keras.layers.Layer) -> tf.keras.layers.Conv2D:
    """Return the final convolutional layer, including inside nested backbones."""
    candidates: list[tf.keras.layers.Conv2D] = []

    def visit(current: tf.keras.layers.Layer) -> None:
        for child in getattr(current, "layers", []):
            visit(child)
            if isinstance(child, tf.keras.layers.Conv2D):
                candidates.append(child)

    visit(layer)
    if not candidates:
        raise ValueError("The loaded model has no Conv2D layer for Grad-CAM.")
    return candidates[-1]


def build_gradcam_model(model: tf.keras.Model) -> tuple[tf.keras.Model, str]:
    """Expose the final convolution activations and softmax output together."""
    last_conv = find_last_conv2d(model)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
        name="gradcam_model",
    )
    return grad_model, last_conv.name


def batch_gradcams(
    grad_model: tf.keras.Model,
    images: tf.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized Grad-CAM maps, predicted labels, and confidence."""
    with tf.GradientTape() as tape:
        conv_output, probabilities = grad_model(images, training=False)
        predicted = tf.argmax(probabilities, axis=1, output_type=tf.int32)
        batch_index = tf.range(tf.shape(probabilities)[0], dtype=tf.int32)
        selected_scores = tf.gather_nd(
            probabilities,
            tf.stack([batch_index, predicted], axis=1),
        )
        objective = tf.reduce_sum(selected_scores)

    gradients = tape.gradient(objective, conv_output)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmaps = tf.reduce_sum(weights * conv_output, axis=-1)
    heatmaps = tf.nn.relu(heatmaps)
    heatmaps /= tf.reduce_max(heatmaps, axis=(1, 2), keepdims=True) + 1e-12
    return (
        heatmaps.numpy(),
        predicted.numpy(),
        tf.reduce_max(probabilities, axis=1).numpy(),
    )


def create_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    """Overlay a resized inferno heatmap on the letterboxed RGB model input."""
    rgb = np.clip(image, 0, 255).astype(np.uint8)
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (rgb.shape[1], rgb.shape[0]),
        Image.Resampling.BILINEAR,
    )
    colour = np.uint8(cm.inferno(np.asarray(heatmap_image) / 255.0)[..., :3] * 255)
    blended = (1.0 - alpha) * rgb + alpha * colour
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))


def write_contact_sheet(
    records: list[dict[str, object]],
    output: Path,
    title: str,
) -> None:
    """Save a small review panel of existing Grad-CAM overlays."""
    if not records:
        return
    columns = 4
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, record in zip(axes, records):
        with Image.open(output.parent.parent / record["overlay_path"]) as image:
            axis.imshow(image)
        predicted = CLASS_NAMES[int(record["predicted_label"])]
        actual = CLASS_NAMES[int(record["true_label"])]
        state = "correct" if record["is_correct"] else "wrong"
        axis.set_title(
            f"true: {actual}\npred: {predicted} ({record['confidence']:.3f})\n{state}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes[len(records) :]:
        axis.axis("off")
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def select_examples(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Select balanced, high-confidence correct and error examples for review."""
    correct: list[dict[str, object]] = []
    for label in range(len(CLASS_NAMES)):
        candidates = [
            record
            for record in records
            if record["is_correct"] and record["true_label"] == label
        ]
        correct.extend(sorted(candidates, key=lambda row: float(row["confidence"]), reverse=True)[:2])

    errors = [record for record in records if not record["is_correct"]]
    # High-confidence errors are particularly informative, but preserve diversity
    # across true classes instead of showing only the largest class.
    error_examples: list[dict[str, object]] = []
    for label in range(len(CLASS_NAMES)):
        candidates = [record for record in errors if record["true_label"] == label]
        error_examples.extend(
            sorted(candidates, key=lambda row: float(row["confidence"]), reverse=True)[:4]
        )
    return correct, error_examples


def run(checkpoint: Path, output: Path, batch_size: int) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output.mkdir(parents=True, exist_ok=True)
    overlay_dir = output / "overlays"
    sheet_dir = output / "contact_sheets"
    overlay_dir.mkdir(exist_ok=True)
    sheet_dir.mkdir(exist_ok=True)

    model = tf.keras.models.load_model(checkpoint, compile=False)
    grad_model, layer_name = build_gradcam_model(model)
    _, _, test = create_transfer_sequences(batch_size=batch_size, seed=42)
    frame = test.frame.reset_index(drop=True)
    if len(frame) != len(test.indices):
        raise RuntimeError("Test manifest and deterministic sequence differ in length.")

    records: list[dict[str, object]] = []
    for batch_number in range(len(test)):
        images, labels = test[batch_number]
        heatmaps, predictions, confidence = batch_gradcams(
            grad_model,
            tf.convert_to_tensor(images, dtype=tf.float32),
        )
        start = batch_number * batch_size
        for offset, (image, heatmap, label_vector, predicted, score) in enumerate(
            zip(images, heatmaps, labels, predictions, confidence)
        ):
            row_index = start + offset
            if row_index >= len(frame):
                break
            true_label = int(np.argmax(label_vector))
            overlay_relative = Path("overlays") / f"{row_index:04d}.jpg"
            create_overlay(image, heatmap).save(output / overlay_relative, quality=92)
            records.append(
                {
                    "row_index": row_index,
                    "image_path": frame.loc[row_index, "image_path"],
                    "true_label": true_label,
                    "true_class": CLASS_NAMES[true_label],
                    "predicted_label": int(predicted),
                    "predicted_class": CLASS_NAMES[int(predicted)],
                    "confidence": float(score),
                    "is_correct": bool(predicted == true_label),
                    "overlay_path": str(overlay_relative),
                }
            )
        print(f"Grad-CAM batch {batch_number + 1}/{len(test)}", flush=True)

    csv_path = output / "gradcam_test_records.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    correct, errors = select_examples(records)
    write_contact_sheet(
        correct,
        sheet_dir / "high_confidence_correct_examples.png",
        "Grad-CAM: high-confidence correct predictions (two per class)",
    )
    write_contact_sheet(
        errors,
        sheet_dir / "high_confidence_error_examples.png",
        "Grad-CAM: high-confidence errors (up to four per true class)",
    )
    summary = {
        "checkpoint": str(checkpoint),
        "test_manifest": str(PROJECT_DIR / "data" / "transfer_final_manifests" / "test.csv"),
        "total_test_images": len(records),
        "correct_predictions": int(sum(bool(record["is_correct"]) for record in records)),
        "incorrect_predictions": int(sum(not bool(record["is_correct"]) for record in records)),
        "last_convolution_layer": layer_name,
        "target_class": "model-predicted class",
        "note": "Post-test explanation only; do not use this output for model selection or tuning.",
    }
    (output / "README.md").write_text(
        "# Grad-CAM test-set diagnostic\n\n"
        + "\n".join(f"- **{key}**: {value}" for key, value in summary.items())
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(records)} Grad-CAM overlays to {output}")
    print(f"Last convolutional layer: {layer_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.checkpoint, arguments.output, arguments.batch_size)
