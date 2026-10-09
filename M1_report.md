# M1 Work Report and M2 Handoff

**Project:** Traffic Density Classification  
**Date:** 06 October 2026  
**Member 1 scope:** Dataset analysis, data cleaning, preprocessing, augmentation, Simple CNN and baseline evaluation.

## 1. Project objective

The project classifies traffic images into five density levels:

| Class | Label |
|---|---:|
| Empty | 0 |
| Low | 1 |
| Medium | 2 |
| High | 3 |
| Traffic Jam | 4 |

The group will compare three approaches:

1. Simple CNN developed by M1.
2. Deep Custom CNN developed by M2.
3. Transfer Learning/Fine-Tuning developed by M3.

All models must use the same cleaned manifests and class order so that their results are comparable.

## 2. Current project structure

```text
dl-traffic-density-classi/
├── Final Dataset/                 # Extracted images, ignored by Git
├── data/
│   └── manifests/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── excluded_images.csv
├── notebooks/
│   ├── 00_data_analysis.ipynb
│   ├── 01_simple_cnn.ipynb
│   ├── 02_complex_cnn.ipynb
│   └── 03_transfer_learning.ipynb
├── scripts/
│   └── build_clean_split.py
├── src/
│   ├── data_loader.py
│   └── evaluation.py
├── ref/                           # Reference material only
├── requirements.txt
└── M1_report.md
```

Everything inside `ref/` is reference material. The notebooks and the main pipeline must not import code from `ref/`.

## 3. Work completed by M1

### 3.1 Dataset setup

The dataset is extracted before running the project. The expected location is:

```text
Final Dataset/
├── training/
├── validation/
└── testing/
```

Each original split contains the five class directories.

The following rule was added to `.gitignore`:

```gitignore
/Final Dataset/
```

This prevents thousands of source images from being committed to Git.

### 3.2 Dataset analysis notebook

File:

```text
notebooks/00_data_analysis.ipynb
```

The notebook performs read-only exploratory analysis:

- Checks the expected split and class directories.
- Scans all images in the extracted dataset.
- Counts images by original split and class.
- Detects empty or unreadable image files.
- Records image width, height and aspect ratio.
- Records file extension, decoded format and color mode.
- Calculates the class imbalance ratio.
- Displays the most common image sizes.
- Plots class distribution and image-size distributions.
- Displays one sample image from each class.

The notebook does not delete, move or modify source images.

### 3.3 Clean split script

File:

```text
scripts/build_clean_split.py
```

The script performs the following steps:

1. Reads images from all three original dataset splits.
2. Converts each readable image to RGB.
3. Computes a SHA-256 hash from its dimensions and decoded RGB pixels.
4. Detects exact duplicate images.
5. Keeps one representative when duplicates have the same class.
6. Excludes the complete duplicate group when the same image has conflicting labels.
7. Combines the remaining images.
8. Creates a new stratified 70/15/15 split using seed 42.
9. Writes clean manifests without changing the original images.

The current implementation detects exact pixel duplicates. It does not yet use perceptual hashing to detect visually similar but non-identical images. This is a documented limitation rather than hidden behaviour.

Run the script from the project root:

```powershell
python .\scripts\build_clean_split.py
```

### 3.4 Clean-data result

The generated manifests currently contain:

| Split | Empty | Low | Medium | High | Traffic Jam | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 925 | 741 | 549 | 311 | 181 | 2,707 |
| Validation | 198 | 158 | 118 | 67 | 39 | 580 |
| Test | 198 | 159 | 118 | 67 | 39 | 581 |
| **Total used** | **1,321** | **1,058** | **785** | **445** | **259** | **3,868** |

Excluded images:

| Reason | Number of images |
|---|---:|
| Exact duplicate | 160 |
| Conflicting label | 10 |
| **Total excluded** | **170** |

The dataset remains imbalanced. `Empty` is the largest class and `Traffic Jam` is the smallest class. Model evaluation should therefore include macro-averaged Precision, Recall and F1-score instead of relying only on Accuracy.

### 3.5 Data loader and augmentation

File:

```text
src/data_loader.py
```

`create_datasets()` reads the three existing manifests. It does not split the data again.

It returns:

```python
train_dataset, val_dataset, test_dataset
```

The class order is fixed as:

```python
[
    "Empty",
    "Low",
    "Medium",
    "High",
    "Traffic Jam",
]
```

Images are decoded by Pillow according to their real contents rather than their filename extensions, then converted to RGB and processed with `tf.image.resize_with_pad()`. The longer side is resized to fit within 224 pixels, while the shorter side is padded to produce a `224 × 224` image without stretching its aspect ratio. Pixel values are then rescaled from `[0, 255]` to `[0, 1]`.

The training dataset can optionally apply:

- Rotation up to ±5 degrees.
- Width and height shift up to 8%.
- Zoom up to 15%.
- Random contrast adjustment.
- Random brightness adjustment.
- Horizontal flip.
- Nearest-pixel filling for empty regions.

Validation and test data are not augmented. Their datasets only load, resize with padding, rescale and batch the images.

The training dataset is shuffled using seed 42. Validation and test datasets keep their manifest order so predictions remain aligned with the true labels.

### 3.6 Shared evaluation function

File:

```text
src/evaluation.py
```

`evaluate_model()` runs `model.predict()` on the test dataset, reads the one-hot labels from the same dataset and returns one table row containing:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Model name | Pending | Pending | Pending | Pending |

Precision, Recall and F1-score use macro averaging by default. Each class therefore contributes equally, including the minority `Traffic Jam` class.

### 3.7 Simple CNN notebook

File:

```text
notebooks/01_simple_cnn.ipynb
```

The notebook follows the presentation style of the instructor's CIFAR-10 example:

1. Import libraries.
2. Set the random seed and experiment settings.
3. Load the clean TensorFlow datasets.
4. Display a batch of traffic images.
5. Present the model architecture.
6. Build the model with Keras `Sequential`.
7. Compile and train the model.
8. Plot training and validation curves.
9. Evaluate the model on the test set.
10. Save the trained model in Keras format.

The proposed architecture is:

```text
Input 224×224×3
        ↓
Conv2D, 16 filters, 3×3, ReLU
        ↓
MaxPooling2D, 2×2
        ↓
Conv2D, 32 filters, 3×3, ReLU
        ↓
MaxPooling2D, 2×2
        ↓
Conv2D, 64 filters, 3×3, ReLU
        ↓
MaxPooling2D, 2×2
        ↓
Flatten
        ↓
Dense, 64 units, ReLU
        ↓
Dense, 5 units, Softmax
```

The notebook uses:

```python
optimizer="adam"
loss="categorical_crossentropy"
metrics=["accuracy"]
```

The baseline setting is:

```python
USE_AUGMENTATION = False
```

The same notebook can be rerun with:

```python
USE_AUGMENTATION = True
```

to compare the same Simple CNN with augmentation.

The model has not yet been trained to completion, so final metrics are not included in this report.

## 4. TensorFlow environment status

The project has been changed from PyTorch to TensorFlow/Keras.

Current `requirements.txt`:

```text
tensorflow
pandas
scikit-learn
pillow
matplotlib
pyyaml
```

`torch` and `torchvision` were removed from the project `.venv`. Installing TensorFlow into that `.venv` was attempted but stopped because the disk did not have enough free space for the 351 MB TensorFlow wheel and its installation files.

For now, use the existing global Python environment that already contains TensorFlow. In VS Code or Jupyter:

1. Open the notebook.
2. Select **Kernel** or **Select Kernel**.
3. Choose the global Python environment containing TensorFlow.
4. Verify the selected environment:

```python
import sys
import tensorflow as tf

print(sys.executable)
print(tf.__version__)
```

The project `.venv` should not be selected until TensorFlow has been installed successfully in it.

## 5. How to load data in a new notebook

Create new notebooks inside:

```text
notebooks/
```

The first code cell should locate the project root and make `src` importable:

```python
import sys
from pathlib import Path

project_dir = Path.cwd()

if not (project_dir / "src").exists():
    project_dir = project_dir.parent

if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))
```

The next cell loads the shared pipeline:

```python
from src.data_loader import CLASS_NAMES, create_datasets
from src.evaluation import evaluate_model
```

Create the datasets:

```python
train_dataset, val_dataset, test_dataset = create_datasets(
    augmentation=True,
    image_size=(224, 224),
    batch_size=32,
)
```

Check that the data and class mapping are correct:

```python
print("Classes:", CLASS_NAMES)
print("Train batches:", train_dataset.cardinality().numpy())
print("Validation batches:", val_dataset.cardinality().numpy())
print("Test batches:", test_dataset.cardinality().numpy())
```

Expected class mapping:

```python
{
    "Empty": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Traffic Jam": 4,
}
```

The dataset labels are converted to one-hot vectors by `tf.one_hot()`. Every model must therefore use a five-unit Softmax output and categorical cross-entropy:

```python
layers.Dense(5, activation="softmax")
```

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

Train with:

```python
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
)
```

Evaluate only after the model configuration has been selected using validation results:

```python
result = evaluate_model(
    model=model,
    test_dataset=test_dataset,
    model_name="Model name",
)

result
```

Important rules:

- Do not run `train_test_split()` again in a model notebook.
- Do not change the class order.
- Do not augment validation or test images.
- Keep validation and test datasets deterministic and unshuffled.
- Use validation results for model selection.
- Use test results only for final comparison.

## 6. Handoff to M2: Deep Custom CNN

M2 owns:

- Deep Custom CNN design.
- Class imbalance experiments.
- Hyperparameter tuning.
- Documentation of the complex architecture and experimental results.

M2 should work in:

```text
notebooks/02_complex_cnn.ipynb
```

### What M2 can reuse

M2 should reuse without changing:

```text
data/manifests/train.csv
data/manifests/val.csv
data/manifests/test.csv
src/data_loader.py
src/evaluation.py
```

This ensures the Simple CNN and Deep Custom CNN use exactly the same data.

### Important loader API changes for M2

The loading pipeline was changed after the first implementation because `ImageDataGenerator` resized every image directly to `224 × 224`, which could distort images with 4:3 or 16:9 aspect ratios.

The current pipeline uses `tf.data.Dataset` and `tf.image.resize_with_pad()` instead. M2 must use the current names below and must not copy the old generator code from earlier notes or chat messages.

| Old name or API | Current name or API | Reason |
|---|---|---|
| `ImageDataGenerator` | `tf.data.Dataset` | Allows aspect-ratio-preserving resize and padding |
| `create_generators()` | `create_datasets()` | The function now returns TensorFlow datasets |
| `train_generator` | `train_dataset` | Updated object type and naming |
| `val_generator` | `val_dataset` | Updated object type and naming |
| `test_generator` | `test_dataset` | Updated object type and naming |
| `flow_from_dataframe()` | `tf.data.Dataset.from_tensor_slices()` | Paths and labels are read directly from manifests |
| `target_size=(224, 224)` | `tf.image.resize_with_pad(image, 224, 224)` | Prevents stretching the image |
| `test_generator.classes` | Labels read from `test_dataset` | `tf.data.Dataset` has no `.classes` attribute |
| `test_generator.reset()` | No replacement needed | The test dataset is deterministic and can be iterated again |

The filenames remain:

```text
src/data_loader.py
src/evaluation.py
```

The clean-split files also remain unchanged:

```text
data/manifests/train.csv
data/manifests/val.csv
data/manifests/test.csv
data/manifests/excluded_images.csv
```

M2 should use the following import:

```python
from src.data_loader import CLASS_NAMES, create_datasets
from src.evaluation import evaluate_model
```

The correct loading call is:

```python
train_dataset, val_dataset, test_dataset = create_datasets(
    augmentation=True,
    image_size=(224, 224),
    batch_size=32,
)
```

The correct training call is:

```python
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
)
```

The correct evaluation call is:

```python
result = evaluate_model(
    model=model,
    test_dataset=test_dataset,
    model_name="Deep Custom CNN",
)
```

Do not use any of the following names in `02_complex_cnn.ipynb`:

```python
create_generators
train_generator
val_generator
test_generator
```

The current preprocessing order is:

```text
Manifest path
→ Pillow decode
→ RGB conversion
→ resize while preserving aspect ratio
→ zero padding to 224×224
→ rescale to [0, 1]
→ augmentation for training only
→ batch
→ prefetch
```

Pillow is retained because the dataset contains a small number of WebP images whose filenames use the `.jpg` extension. Decoding by content prevents these files from failing in the TensorFlow pipeline.

### Suggested M2 notebook flow

1. Select the global TensorFlow environment.
2. Add the project root to `sys.path`.
3. Load datasets using `create_datasets()`.
4. Display the data counts and class mapping.
5. Define reusable CNN blocks inside the notebook.
6. Build the Deep Custom CNN.
7. Print `model.summary()` and describe each block.
8. Compile and train using the validation dataset.
9. Plot loss and accuracy.
10. Evaluate once on the test dataset.
11. Add the result to the shared model comparison table.

A suitable custom block can conceptually contain:

```text
Conv2D
→ BatchNormalization
→ ReLU
→ Conv2D
→ BatchNormalization
→ ReLU
→ MaxPooling2D
```

The final M2 architecture should be deeper than the Simple CNN but must still be designed by the group rather than copied from a pretrained architecture.

### Class imbalance experiments

M2 should compare clearly defined alternatives instead of changing several factors at once. Examples:

- Normal categorical cross-entropy.
- Categorical cross-entropy with class weights.
- Augmentation disabled versus enabled.

When using class weights, calculate them only from the training manifest. Do not use validation or test labels to calculate training weights.

### Hyperparameter tuning

Possible parameters:

- Number of filters.
- Number of CNN blocks.
- Dense-layer size.
- Dropout rate.
- Learning rate.
- Batch size.

Change a small number of parameters per experiment and record every configuration. Select the final model using validation performance, preferably macro F1 or validation accuracy combined with per-class behaviour.

### Required M2 outputs

- Architecture diagram or clear text flow.
- `model.summary()`.
- Training and validation curves.
- Final Accuracy, macro Precision, macro Recall and macro F1-score.
- Comparison with the M1 Simple CNN.
- Discussion of overfitting and minority-class performance.

## 7. Pending work and known limitations

- TensorFlow is not installed in the project `.venv`; use the global TensorFlow kernel temporarily.
- The current README still contains old ZIP/PyTorch instructions and must be updated before final submission.
- The current clean-split script detects exact duplicates only, not resized or visually near-duplicate images.
- Simple CNN training and final metrics are still pending.
- Confusion matrix and per-class result visualisation have not yet been added to `evaluation.py`.
- `02_complex_cnn.ipynb` and `03_transfer_learning.ipynb` are currently empty placeholders.

## 8. Reproducibility settings

Current shared settings:

```text
Seed: 42
Image size: 224 × 224
Default batch size: 32
Train/validation/test ratio: approximately 70/15/15
Number of classes: 5
Label mode: categorical one-hot
```

All members should keep these settings unless a change is explicitly recorded as an experiment.
