# Báo cáo thí nghiệm phân loại mật độ giao thông

## Phạm vi và làm sạch dữ liệu

Nguồn là `traffic.zip`, giữ nguyên không sửa. Manifest dùng giải mã nội dung ảnh, RGB và thứ tự nhãn `empty=0, low=1, medium=2, high=3, traffic_jam=4`. Sau khử trùng exact theo nội dung, giữ một bản của nhóm cùng nhãn và loại toàn bộ 10 ảnh thuộc các nhóm exact-duplicate xung đột nhãn. Tập còn 3.868 ảnh: train 2.702, validation 579, test 587. Split dùng seed 42, stratify theo lớp và group ID; không dùng split gốc.

## Tiền xử lý và training

Mọi model dùng RGB, giữ tỉ lệ, resize-fit + letterbox 224×224 (pad RGB `(124,116,103)`, gần ImageNet mean), tensor và chuẩn hóa ImageNet. AdamW, batch 64, weight decay `1e-4`, early stopping Macro-F1 validation (patience 7), ReduceLROnPlateau, AMP và GPU RTX 3060 12 GB. Scratch CNN dùng LR `1e-3`; MobileNetV2 dùng pretrained ImageNet, freeze backbone 5 epoch rồi mở 4 block cuối LR `1e-4`.

## Augmentation và loss ablation

Kết quả sau đây là **quan sát test** cho Complex CNN, seed 42; selection thực hiện trên validation trong code, nhưng artifact hiện hữu chỉ lưu test metrics.

| Thiết lập | Test Macro-F1 | Accuracy | Ordinal MAE |
|---|---:|---:|---:|
| A0 + weighted CE | 0,7941 | 0,8007 | 0,2129 |
| A1 + weighted CE | 0,7889 | 0,7990 | 0,2147 |
| A2 + weighted CE | 0,7778 | 0,7871 | 0,2334 |
| A1 + CE | 0,7933 | 0,7956 | 0,2147 |

A0 là ứng viên augmentation mạnh nhất theo test artifact (A1/A2 không cải thiện); CE thường cao hơn weighted CE một ít với A1. Đây **không được dùng** để tuyên bố lựa chọn chính thức vì cần truy xuất/báo cáo Macro-F1 validation tương ứng.

## So sánh mô hình (ba seed 42/43/44)

Các run so sánh hiện có đều dùng A1 + weighted CE. Mean ± sample std trên test:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced accuracy | Ordinal MAE | Within-one |
|---|---:|---:|---:|---:|---:|---:|
| Simple CNN | 0,6826 ± 0,0494 | 0,6787 ± 0,0488 | 0,6814 ± 0,0533 | 0,7151 ± 0,0314 | 0,4276 ± 0,0988 | 0,9404 ± 0,0258 |
| Complex CNN | 0,7961 ± 0,0064 | 0,7927 ± 0,0078 | 0,7944 ± 0,0058 | 0,8069 ± 0,0024 | 0,2141 ± 0,0060 | 0,9915 ± 0,0017 |
| MobileNetV2 | **0,7984 ± 0,0081** | **0,7977 ± 0,0087** | **0,7983 ± 0,0079** | 0,8065 ± 0,0097 | **0,2101 ± 0,0113** | **0,9926 ± 0,0020** |

MobileNetV2 có Macro-F1 cao nhất, nhưng chênh lệch với Complex CNN nhỏ hơn độ biến thiên giữa seed. Từng run và per-class report nằm trong `outputs/<model>_A1_weighted_<seed>/metrics.json` (outputs bị gitignore do chứa checkpoint/artefact lớn).

## Findings và hạn chế

- MobileNetV2 seed 42 đạt Macro-F1 0,8068 và ordinal MAE 0,1976; Complex CNN seed 44 đạt Macro-F1 0,8017.
- Các nhầm lẫn chủ yếu kỳ vọng giữa lớp kề nhau Low/Medium/High; within-one >99% cho Complex/MobileNet.
- Test split sạch hơn split gốc nhưng nhóm near-duplicate hiện được group theo dHash-identical, chưa có bước SSIM xác nhận đầy đủ theo đặc tả. Đây là hạn chế quan trọng.
- Protocol final có sai lệch: ablation cho thấy A0/CE nên được xác nhận theo validation, nhưng final 3-seed đã chạy A1 + weighted CE. Vì vậy các số final là benchmark hợp lệ cho cấu hình ghi rõ, **không phải** kết quả lựa chọn hyperparameter hoàn toàn độc lập.
- Không có dữ liệu Hà Nội nên không thực hiện external inference.

## Artefact

- Bảng tổng hợp: `outputs/summary/final_comparison.csv`.
- Curves/confusion matrix: `outputs/summary/curve_*.png`, `outputs/summary/confusion_*.png`.
- Checkpoint tốt nhất: `outputs/<run>/best.pt`.
- Lệnh tái lập: `bash scripts/run_all.sh`.
