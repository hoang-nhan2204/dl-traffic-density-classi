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

## Chú ý 
mọi file ở trong folder ref là tài liệu tham khảo, không nằm trong pipeline chạy chính của dự án 