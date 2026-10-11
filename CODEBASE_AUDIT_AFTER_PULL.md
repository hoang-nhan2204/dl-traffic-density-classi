# Audit codebase sau pull

## Current repo structure

- Branch hiện tại: `main`, đồng bộ `origin/main` tại commit `7206599` (`notebooks0`); working tree sạch tại thời điểm audit.
- Pipeline active: `scripts/build_clean_split.py`, `src/data_loader.py`, `src/evaluation.py`, và `notebooks/`.
- `Final Dataset/` và `data/manifests/` **không tồn tại trong working tree hiện tại**. Cả hai đều bị Git ignore; vì vậy pipeline active chưa thể chạy sau clone/pull nếu không được cung cấp dữ liệu và manifest.
- `outputs/`, `.venv/`, `traffic.zip` là local/ignored. `README.md`, `M1_report.md`, `DATASET_AUDIT.md`, `EXPERIMENT_REPORT.md` cùng tồn tại nhưng không cùng một pipeline.

## Active vs old/reference code

**Active theo commit mới và `M1_report.md`:**

- `scripts/build_clean_split.py`: tạo manifest từ ảnh đã giải nén.
- `src/data_loader.py`: TensorFlow/Keras `ImageDataGenerator` chung.
- `src/evaluation.py`: evaluation macro metrics cơ bản.
- `notebooks/00_data_analysis.ipynb`: EDA dữ liệu gốc.
- `notebooks/01_simple_cnn.ipynb`: M1 Simple CNN, có cell đã chạy.

**Old/reference:** toàn bộ `ref/`. Đây là pipeline PyTorch cũ (`ref/scripts/build_manifest.py`, `ref/src/dataset.py`, `ref/src/transforms.py`, `ref/scripts/train.py`) đọc trực tiếp `traffic.zip`, có letterbox/ImageNet normalization và các kết quả cũ. Không có import active nào tham chiếu `ref/`; không dùng nó làm nguồn chân lý hiện tại.

**Không hoàn chỉnh:** `notebooks/02_complex_cnn.ipynb` và `notebooks/03_transfer_learning.ipynb` là file rỗng 0 byte. Do đó hiện chưa có implementation active cho M2/M3.

## Current data pipeline

Luồng code active thực tế là:

`Final Dataset/{training,validation,testing}/{Empty,Low,Medium,High,Traffic Jam}`
→ `scripts/build_clean_split.py:scan_images()`
→ `calculate_hash()` (SHA-256 của kích thước + RGB pixels)
→ `remove_duplicates()`
→ `create_splits()`
→ `data/manifests/{train,val,test}.csv`
→ `src/data_loader.py:create_generators()`
→ TensorFlow Keras generators trong notebook.

- `scan_images()` mở ảnh bằng Pillow, gọi `image.load()`, hash sau `convert("RGB")`; ảnh lỗi bị ghi vào `excluded_images.csv`.
- `remove_duplicates()` giữ 1 đại diện của duplicate cùng nhãn và loại toàn bộ nhóm duplicate khác nhãn.
- `create_splits()` dùng `train_test_split(..., stratify=label, random_state=42)` hai lần, tỷ lệ 70/15/15. Nó không giữ original train/validation/test split, nên mục tiêu là clean split mới.
- Class mapping trong `scripts/build_clean_split.py:CLASS_TO_LABEL` là chính xác: `Empty=0, Low=1, Medium=2, High=3, Traffic Jam=4`.
- CSV kỳ vọng cột: `image_path`, `label`, `class_name`, `split`, `original_split`. `data_loader.py` dùng `image_path` và `class_name`; `label` không được đọc trực tiếp.
- `create_generators()` đặt `classes=CLASS_NAMES` theo đúng thứ tự chuẩn, `color_mode="rgb"`, `target_size=(224,224)`, `rescale=1/255`.

Những điều **không** được active code bảo đảm:

- Không có near-duplicate dHash/SSIM grouping; chỉ hash pixel exact.
- `ImageDataGenerator(target_size=...)` resize trực tiếp về 224×224, **không** giữ aspect ratio/letterbox.
- Không dùng ImageNet normalization; chỉ `/255`.
- Manifest chưa được lưu/track trong repo hiện tại và không hiện diện local.

## Data contract for model notebooks

Interface hiện có duy nhất là:

```python
from src.data_loader import CLASS_NAMES, create_generators

train_generator, val_generator, test_generator = create_generators(
    augmentation=..., image_size=(224, 224), batch_size=32,
)
```

- Điều kiện trước: Member 1 đã giải nén dataset vào `Final Dataset/` và đã chạy `python scripts/build_clean_split.py` từ repo root.
- `train_generator`: shuffle=True, categorical one-hot labels, optional augmentation nội bộ.
- `val_generator`, `test_generator`: shuffle=False, RGB, direct resize, `/255`, không augmentation.
- Paths trong manifest là relative từ project root, ví dụ `Final Dataset/training/...`.
- Dataset objects/DataLoaders cho PyTorch không có trong active code. Các notebook mới hiện phải dùng Keras generators hoặc đội nhóm phải thống nhất data interface khác trước khi triển khai.

Không tự tạo split, tự map label, hoặc đọc original split trong notebook M2/M3. Không tự thêm preprocessing mâu thuẫn; preprocessing đang xảy ra **bên trong** `create_generators()`, còn augmentation train cũng đang xảy ra ở đây.

## Existing model/notebook implementations

| Artifact active | Nội dung hiện có | Dữ liệu / preprocessing | Train/evaluate/output |
|---|---|---|---|
| `notebooks/00_data_analysis.ipynb` | EDA read-only: cấu trúc, decode, format/mode, size/aspect, imbalance, samples | Đọc `Final Dataset/`; chuyển RGB chỉ lúc hiển thị | Không train, không checkpoint |
| `notebooks/01_simple_cnn.ipynb` | Keras Sequential: Conv 16/32/64 + pool, Flatten, Dense 64, softmax 5 | Gọi `create_generators()`; baseline `USE_AUGMENTATION=False` | Adam + categorical crossentropy, accuracy, 10 epoch; `evaluate_model()` macro P/R/F1; ghi `outputs/simple_cnn.keras` |
| `notebooks/02_complex_cnn.ipynb` | Rỗng | Không có | Không có |
| `notebooks/03_transfer_learning.ipynb` | Rỗng | Không có | Không có |
| `src/evaluation.py:evaluate_model()` | Accuracy, macro precision/recall/F1 | `test_generator.reset()`, `model.predict()` | Trả `DataFrame`; không confusion matrix/per-class/ordinal/checkpoint management |

Logic reference có thể tham khảo về manifest, letterbox/ImageNet normalization, metrics/checkpoint trong `ref/`, nhưng là PyTorch và không tương thích trực tiếp với active TensorFlow pipeline.

## Methodology inconsistencies or risks

1. **Tài liệu mâu thuẫn:** `README.md` vẫn mô tả `traffic.zip`, `scripts/build_manifest.py` và PyTorch CUDA, nhưng file active đó không tồn tại ngoài `ref/`; pipeline active dùng dataset đã giải nén + TensorFlow.
2. **Tiền xử lý mâu thuẫn:** `ref/src/transforms.py` dùng aspect-preserving letterbox + ImageNet normalization; `src/data_loader.py` dùng direct stretch + `/255`.
3. **Augmentation active** (`rotation`, shift 8%, shear 5, zoom 0.85–1.15, brightness 0.6–1.4) có thể khác setting baseline trong `01_simple_cnn.ipynb` và chưa được khóa thành protocol nhóm.
4. **Loss:** M1 dùng unweighted categorical crossentropy; class imbalance tồn tại. M2/M3 không có code nên chưa biết có giữ cùng loss hay không.
5. **So sánh chưa sẵn sàng:** Không có M2/M3 notebook active, no shared callback/seed/epoch protocol, metrics evaluation còn giới hạn và chưa có per-class/confusion matrix.
6. **Leakage:** active cleaning chỉ xử lý exact decoded-pixel duplicate, không xác nhận near duplicate; audit trước đó đã phát hiện near-duplicate risk.

## Reproducibility issues

- `requirements.txt` là TensorFlow stack, nhưng `M1_report.md` nói phải dùng global Python vì `.venv` thiếu TensorFlow; không có `environment.yml`, lockfile hoặc kernel spec trỏ tới environment đó.
- Notebook kernelspec chỉ là generic `Python 3`; không khóa executable/version.
- Đường dẫn source là relative qua `Path(__file__)`/`Path.cwd()`, không có absolute hard-code trong active source; đây là điểm tốt.
- Seed 42 có trong split, generator và Simple CNN. Tuy nhiên không thấy deterministic TensorFlow/GPU setup đầy đủ.
- `outputs/` và manifests bị ignore; teammate clone repo không có dữ liệu để reproduce mà không nhận handoff ngoài Git.

## What my notebook should consume

Sau khi Member 1 cung cấp dữ liệu và đã tạo manifest, notebook của bạn nên import **chỉ**:

```python
from src.data_loader import CLASS_NAMES, create_generators
```

và gọi `create_generators()` với cùng `image_size=(224, 224)`, `batch_size` đã thống nhất, augmentation policy đã thống nhất. Giữ `CLASS_NAMES` theo thứ tự `["Empty", "Low", "Medium", "High", "Traffic Jam"]`, dùng train/val/test trả về và chỉ tune trên validation; test chỉ dùng khi chốt model.

Có thể reuse `src/evaluation.py:evaluate_model()` cho bảng macro metrics. Không reuse `ref/` trực tiếp, không tự đọc `traffic.zip`, không viết loader/split mới, và không xử lý resize/normalization lần thứ hai trong model notebook.

## Decisions still needed

1. Team phải chốt TensorFlow/Keras là active stack hay quay về PyTorch; hiện docs/reference mâu thuẫn.
2. Member 1 phải publish/cung cấp một bộ manifest canonical cùng dữ liệu giải nén, checksum/version và số lượng cuối cùng.
3. Chốt resize policy (direct stretch hiện tại hay letterbox), normalization (`/255` hay ImageNet), augmentation, loss/class weights, seeds/epochs/early stopping và metrics bắt buộc.
4. Quyết định có mở rộng cleaning bằng near-duplicate confirmation trước khi manifest được đóng băng không.
5. M2/M3 cần tạo notebook hợp lệ từ template chung trước training.

## Recommended next step

**Chưa sẵn sàng để bắt đầu notebook model có thể so sánh công bằng.** Trước hết đọc theo thứ tự: `M1_report.md` → `scripts/build_clean_split.py` → `src/data_loader.py` → `notebooks/01_simple_cnn.ipynb`. Sau đó yêu cầu Member 1 handoff `Final Dataset/` cùng `data/manifests/*.csv`, và cả nhóm chốt một shared preprocessing/training protocol bằng văn bản. Khi đó notebook M2 có thể dùng trực tiếp `create_generators()` thay vì tái triển khai data layer.
