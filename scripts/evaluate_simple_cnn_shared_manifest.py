"""Evaluate legacy PyTorch Simple CNN checkpoints on the final shared manifest.

The evaluation is inference-only: no checkpoint, manifest, split, or model
weights are changed.  It preserves the checkpoint's original RGB letterbox and
ImageNet normalization while replacing only the legacy test manifest with the
shared 574-image test manifest.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tvf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref"))
from src.models import SimpleCNN  # noqa: E402


CLASS_NAMES = ["Empty", "Low", "Medium", "High", "Traffic Jam"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MANIFEST = ROOT / "data" / "transfer_final_manifests" / "test.csv"
ARCHIVE = ROOT / "traffic.zip"
OUTPUT_ROOT = ROOT / "outputs" / "simple_cnn_shared_manifest"


def letterbox(image: Image.Image, size: int = 224) -> Image.Image:
    """The exact spatial preprocessing used by the PyTorch Simple CNN."""
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (124, 116, 103))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


class SharedManifestDataset(Dataset):
    def __init__(self, manifest: Path, archive: Path):
        self.frame = pd.read_csv(manifest)
        self.archive = archive
        self._zip: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive)
        row = self.frame.iloc[index]
        with Image.open(io.BytesIO(self._zip.read(row.image_path))) as image:
            processed = letterbox(image)
        tensor = tvf.normalize(tvf.to_tensor(processed), MEAN, STD)
        return tensor, int(row.label), str(row.image_path)


def safe_load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Load only tensor/primitive checkpoint contents; never unpickle arbitrary code."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Expected a state-dict checkpoint with a 'model' key: {checkpoint}")
    state = payload["model"]
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"Checkpoint model entry is not a pure tensor state dict: {checkpoint}")
    return state


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ordinal_mae": float(np.abs(y_true - y_pred).mean()),
        "within_one_accuracy": float((np.abs(y_true - y_pred) <= 1).mean()),
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(CLASS_NAMES))
        },
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix(matrix: list[list[int]], output: Path, title: str) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate(seed: int, batch_size: int, device: torch.device) -> dict[str, object]:
    checkpoint = ROOT / "outputs" / f"simple_A1_weighted_{seed}" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output = OUTPUT_ROOT / f"simple_A1_weighted_s{seed}"
    output.mkdir(parents=True, exist_ok=True)

    model = SimpleCNN().to(device)
    model.load_state_dict(safe_load_state_dict(checkpoint), strict=True)
    model.eval()
    dataset = SharedManifestDataset(MANIFEST, ARCHIVE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    confidences: list[float] = []
    paths: list[str] = []
    with torch.inference_mode():
        for images, labels, batch_paths in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            paths.extend(batch_paths)

    y_true = np.asarray(true_labels)
    y_pred = np.asarray(predicted_labels)
    result = metrics(y_true, y_pred)
    result.update(
        {
            "experiment_id": f"simple_A1_weighted_s{seed}",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "test_manifest": str(MANIFEST),
            "test_samples": int(len(y_true)),
            "params": int(sum(parameter.numel() for parameter in model.parameters())),
            "preprocessing": {
                "colour": "RGB",
                "resize": "aspect-ratio-preserving letterbox 224x224",
                "padding_rgb": [124, 116, 103],
                "normalization": {"mean": list(MEAN), "std": list(STD)},
                "test_augmentation": False,
            },
            "note": "Inference-only evaluation of the existing legacy checkpoint on the final shared test manifest.",
        }
    )
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savetxt(output / "confusion_matrix.csv", result["confusion_matrix"], delimiter=",", fmt="%d")
    save_confusion_matrix(
        result["confusion_matrix"],
        output / "confusion_matrix.png",
        f"Simple CNN A1 + weighted CE (seed {seed})\nShared 574-image test manifest",
    )
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "true_label", "true_class", "predicted_label", "predicted_class", "confidence"],
        )
        writer.writeheader()
        for path, truth, prediction, confidence in zip(paths, y_true, y_pred, confidences):
            writer.writerow(
                {
                    "image_path": path,
                    "true_label": int(truth),
                    "true_class": CLASS_NAMES[int(truth)],
                    "predicted_label": int(prediction),
                    "predicted_class": CLASS_NAMES[int(prediction)],
                    "confidence": float(confidence),
                }
            )
    print(json.dumps({key: result[key] for key in ("experiment_id", "accuracy", "macro_f1")}), flush=True)
    return result


def write_summary(results: list[dict[str, object]]) -> None:
    columns = [
        "experiment_id", "seed", "test_samples", "params", "accuracy", "macro_precision",
        "macro_recall", "macro_f1", "weighted_f1", "balanced_accuracy", "ordinal_mae",
        "within_one_accuracy",
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: result[column] for column in columns} for result in results])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {runtime_device}", flush=True)
    completed = [evaluate(seed, args.batch_size, runtime_device) for seed in args.seeds]
    write_summary(completed)
