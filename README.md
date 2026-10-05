# Traffic Density Classification

Dự án so sánh Simple CNN, Complex CNN và MobileNetV2 cho năm mức mật độ: `empty`, `low`, `medium`, `high`, `traffic_jam`.

## Cài đặt

```bash
uv venv .venv
uv pip install --python .venv/bin/python --torch-backend=cu128 -r requirements.txt
```

Đặt `traffic.zip` tại thư mục gốc (file không được commit). Cần GPU NVIDIA để tái lập tốc độ chạy đã báo cáo.

## Tạo clean split

```bash
.venv/bin/python scripts/build_manifest.py
```

Lệnh tạo `data/manifests/{train,val,test}.csv`, dùng seed 42, split xấp xỉ 70/15/15, khử duplicate exact và group theo candidate dHash. Không thay đổi ZIP gốc.

## Huấn luyện

```bash
.venv/bin/python scripts/train.py --model complex --aug A1 --loss weighted --seed 42 --epochs 40
bash scripts/run_all.sh
```

`run_all.sh` chạy screening A0/A1/A2, CE/weighted CE, sau đó ba seed cho mỗi kiến trúc. Kết quả của mỗi run là `outputs/<model>_<aug>_<loss>_<seed>/`, bao gồm `best.pt`, `metrics.json`, `history.json`, `predictions.csv`.

## Tổng hợp

```bash
.venv/bin/python scripts/summarize_results.py
```

Tạo bảng mean ± std, curve validation và confusion matrix trong `outputs/summary/`. Xem [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) để biết kết quả và giới hạn protocol.

## Thiết kế

Tiền xử lý chung: decode nội dung → RGB → resize-fit giữ tỷ lệ → pad 224×224 → ImageNet normalization. `src/models/nets.py` có ba kiến trúc; `src/dataset.py` đọc trực tiếp từ ZIP. Mã/artefact tái lập dùng đường dẫn tương đối.
