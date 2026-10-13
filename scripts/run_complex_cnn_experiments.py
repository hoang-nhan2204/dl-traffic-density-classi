"""Run the three M2 Complex-CNN ablations on the shared clean manifests.

This mirrors the architecture and experiment matrix in
``notebooks/02_complex_cnn.ipynb``.  It intentionally does not execute notebook
cells which inspect/repair source files.  Test is evaluated once, only after
the validation-selected configuration is frozen.
"""
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.transfer_data import CLASS_NAMES, ManifestSequence

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "complex_cnn"
SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 20


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


class CachedSequence(tf.keras.utils.Sequence):
    """Cache immutable RGB letterbox data in RAM; augmentation stays in-model."""

    def __init__(self, split, batch_size, shuffle, seed):
        super().__init__()
        source = ManifestSequence(split, batch_size=1, shuffle=False, seed=seed)
        self.frame = source.frame.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.images = np.empty((len(self.frame), *IMAGE_SIZE, 3), dtype=np.float32)
        self.labels = self.frame["label"].to_numpy(dtype=np.int32)
        for index in range(len(source)):
            image, _ = source[index]
            self.images[index] = image[0] / 255.0
        source.archive.close()
        self.indices = np.arange(len(self.frame))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        return self.images[ids], tf.keras.utils.to_categorical(self.labels[ids], 5)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def build_deep_cnn(augmentation):
    layers = tf.keras.layers
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="rgb_letterbox")
    x = inputs
    if augmentation:
        # Same augmentation family/intensities as src.data_loader; train-only.
        x = tf.keras.Sequential([
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(5 / 360, fill_mode="nearest", seed=SEED),
            layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=SEED),
            layers.RandomZoom(0.15, 0.15, fill_mode="nearest", seed=SEED),
            layers.RandomContrast(0.2, seed=SEED),
            layers.RandomBrightness(0.2, value_range=(0, 1), seed=SEED),
        ], name="train_augmentation")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(5, activation="softmax", dtype="float32", name="density")(x)
    return tf.keras.Model(inputs, outputs, name="deep_custom_cnn")


class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.dataset.labels
        y_pred = self.model.predict(self.dataset, verbose=0).argmax(axis=1)
        logs["val_macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)


def make_callbacks(run_dir, val):
    return [
        ValidationMacroF1(val),
        tf.keras.callbacks.ModelCheckpoint(
            run_dir / "best_val_loss.keras", monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, start_from_epoch=5
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(run_dir / "epochs.csv"),
    ]


def validation_metrics(model, dataset):
    y_true = dataset.labels
    y_pred = model.predict(dataset, verbose=0).argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(5), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_history(history, run_dir):
    frame = pd.DataFrame(history.history)
    frame.to_csv(run_dir / "history.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metric, axis, title in [("loss", axes[0], "Loss"), ("accuracy", axes[1], "Accuracy")]:
        axis.plot(frame[metric], label="train")
        axis.plot(frame[f"val_{metric}"], label="validation")
        axis.set(title=title, xlabel="Epoch", ylabel=metric)
        axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "curves.png", dpi=160)
    plt.close(figure)


def run_experiment(spec, train, val, class_weights):
    run_dir = OUT / spec["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "completed.json"
    if done.exists():
        return json.loads(done.read_text())
    set_seed()
    start = time.time()
    model = build_deep_cnn(spec["augmentation"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    history = model.fit(
        train, validation_data=val, epochs=EPOCHS,
        class_weight=class_weights if spec["class_weights"] else None,
        callbacks=make_callbacks(run_dir, val), verbose=2,
    )
    metrics = validation_metrics(model, val)
    plot_history(history, run_dir)
    model.save(run_dir / "model.keras")
    result = {
        **spec, **metrics, "params": int(model.count_params()),
        "epochs_trained": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "lowest_val_loss": float(min(history.history["val_loss"])),
        "seconds": time.time() - start,
        "model_path": str(run_dir / "model.keras"),
    }
    done.write_text(json.dumps(result, indent=2))
    tf.keras.backend.clear_session(); gc.collect()
    return result


def test_metrics(model_path, test):
    model = tf.keras.models.load_model(model_path, compile=False)
    y_true = test.labels
    y_pred = model.predict(test, verbose=0).argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(5), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=range(5))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {CLASS_NAMES[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                        "f1": float(f1[i]), "support": int(support[i])} for i in range(5)},
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"GPU: {gpus}; policy: {tf.keras.mixed_precision.global_policy().name}; batch: {BATCH_SIZE}")
    print("Caching RGB letterbox images in RAM once for all experiments...")
    train = CachedSequence("train", BATCH_SIZE, shuffle=True, seed=SEED)
    val = CachedSequence("val", BATCH_SIZE, shuffle=False, seed=SEED)
    test = CachedSequence("test", BATCH_SIZE, shuffle=False, seed=SEED)
    counts = np.bincount(train.labels, minlength=5)
    class_weights = {i: len(train.labels) / (5 * count) for i, count in enumerate(counts)}
    specs = [
        {"id": "exp1_baseline", "name": "Deep CNN baseline", "augmentation": False, "class_weights": False},
        {"id": "exp2_augmentation", "name": "Deep CNN + Augmentation", "augmentation": True, "class_weights": False},
        {"id": "exp3_augmentation_weights", "name": "Deep CNN + Augmentation + Class Weights", "augmentation": True, "class_weights": True},
    ]
    results = [run_experiment(spec, train, val, class_weights) for spec in specs]
    table = pd.DataFrame(results).sort_values(
        ["macro_f1", "accuracy", "lowest_val_loss"], ascending=[False, False, True]
    )
    table.to_csv(OUT / "experiment_registry.csv", index=False)
    selected = table.iloc[0].to_dict()
    test_file = OUT / "test_metrics.json"
    if test_file.exists():
        test_result = json.loads(test_file.read_text())
    else:
        test_result = test_metrics(selected["model_path"], test)
        test_result["selected_experiment"] = selected["name"]
        test_file.write_text(json.dumps(test_result, indent=2))
        np.savetxt(OUT / "test_confusion_matrix.csv", test_result["confusion_matrix"], delimiter=",", fmt="%d")
    (OUT / "selected_config.json").write_text(json.dumps(selected, indent=2))
    print("SELECTED", selected["name"])
    print(json.dumps({key: test_result[key] for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
