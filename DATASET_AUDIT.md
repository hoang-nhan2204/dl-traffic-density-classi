# Kiểm toán bộ dữ liệu phân loại mật độ giao thông

Nguồn đã kiểm tra: `traffic.zip` (232.976.918 byte; xấp xỉ 222,2 MiB). Thời điểm kiểm toán: 05-10-2026. Toàn bộ 4.038 tệp ảnh được giải mã trực tiếp từ ZIP bằng Pillow; không huấn luyện mô hình và không sửa đổi dữ liệu nguồn.

## 1. Tóm tắt điều hành

Bộ dữ liệu có cấu trúc ba split hợp lệ và năm nhãn nhất quán: `Empty`, `Low`, `Medium`, `High`, `Traffic Jam`. Không có ảnh hỏng, tệp 0 byte hoặc tệp không-phải-ảnh trong thư mục lớp. Tuy nhiên, split hiện tại **không an toàn để đánh giá mô hình**: có 105 nhóm ảnh byte-giống-hệt xuất hiện qua split (120 cặp cross-split), trong đó có hai nhóm mang nhãn mâu thuẫn; thêm 895 cặp ứng viên gần-trùng qua split theo dHash. Cần làm sạch/tạo lại split theo nhóm nguồn trước huấn luyện.

## 2. Cấu trúc dữ liệu

```
Final Dataset/
├── training/{Empty, High, Low, Medium, Traffic Jam}/
├── validation/{Empty, High, Low, Medium, Traffic Jam}/
└── testing/{Empty, High, Low, Medium, Traffic Jam}/
```

Cả ba split đều có đủ đúng năm lớp, không có thư mục lớp dư. ZIP có 4.038 entry và tất cả đều là tệp ảnh nằm trong cấu trúc trên.

## 3. Thống kê split và lớp

| Split | Lớp | Số ảnh | Tỷ lệ trong split |
|---|---:|---:|---:|
| testing | Empty | 64 | 20,00% |
| testing | High | 64 | 20,00% |
| testing | Low | 64 | 20,00% |
| testing | Medium | 64 | 20,00% |
| testing | Traffic Jam | 64 | 20,00% |
| training | Empty | 1.186 | 35,11% |
| training | High | 378 | 11,19% |
| training | Low | 936 | 27,71% |
| training | Medium | 688 | 20,37% |
| training | Traffic Jam | 190 | 5,62% |
| validation | Empty | 120 | 35,29% |
| validation | High | 38 | 11,18% |
| validation | Low | 94 | 27,65% |
| validation | Medium | 70 | 20,59% |
| validation | Traffic Jam | 18 | 5,29% |

Tổng theo lớp: Empty 1.370 (33,93%), High 480 (11,89%), Low 1.094 (27,09%), Medium 822 (20,36%), Traffic Jam 272 (6,74%). Tổng split: training 3.378, validation 340, testing 320.

Tỉ số lớp lớn nhất/nhỏ nhất là 6,24 ở training (1.186/190), 6,67 ở validation (120/18) và 5,04 toàn bộ dữ liệu. Training và validation có phân bố gần như trùng khớp, nên có dấu hiệu được chia phân tầng; test được cân bằng chủ đích 64 ảnh/lớp, không cùng phân bố với train/validation.

## 4. Định dạng, khả năng đọc và kênh màu

| Thuộc tính | Kết quả |
|---|---|
| Phần mở rộng | `.jpg`: 4.027; `.jpeg`: 11 |
| Định dạng sau giải mã | JPEG: 4.033; WEBP: 4; PNG: 1 |
| Chế độ màu | RGB: 4.037; RGBA: 1 |
| Ảnh lỗi/không đọc được | 0 |
| Tệp 0 byte | 0 |
| Tệp không phải ảnh | 0 |

Năm tệp mang hậu tố `.jpg` nhưng định dạng thật không phải JPEG: `testing/Traffic Jam/20210125_DLI-AKS-MN_Ghazipur_border-16_1611613688877_1611613697329.jpg`, `testing/Traffic Jam/826749-jam.jpg`, `testing/Traffic Jam/904237-untitled-design.jpg`, `training/Traffic Jam/79122305.jpg`, và `validation/Traffic Jam/1604126519_untitled-design.jpg`. Pipeline phải giải mã theo nội dung (Pillow/OpenCV), không dựa vào hậu tố; bước `convert('RGB')` xử lý luôn ảnh RGBA duy nhất.

Có 106 nhóm tên tệp bị lặp. Phần lớn tên (3.901/4.038) giống UUID; 7 ảnh có EXIF. Tên UUID không cung cấp camera, thời điểm hay chuỗi video; các tên không-UUID tập trung ở lớp Traffic Jam và mang dáng dấp ảnh web/biên tập. Không có đủ metadata để chứng minh nguồn camera hay thời gian cho từng ảnh.

## 5. Kích thước và tỷ lệ khung hình

| Chỉ số | Rộng | Cao |
|---|---:|---:|
| Nhỏ nhất | 111 | 119 |
| Q1 | 640 | 360 |
| Trung bình | 611,53 | 403,13 |
| Trung vị | 640 | 360 |
| Q3 | 640 | 480 |
| Lớn nhất | 4.256 | 2.832 |

Có 67 độ phân giải khác nhau. Phổ biến nhất là 640×360 (1.796), 640×480 (1.777), và 320×240 (363): ba độ phân giải này chiếm 97,5% ảnh. Tỷ lệ rộng/cao: min 0,714; Q1 1,333; trung vị 1,333; Q3 1,778; trung bình 1,540; max 2,977.

- Cạnh bất kỳ <64 px: 0 (0,00%); <128 px: 1 (0,02%); <224 px: 58 (1,44%).
- Cả hai cạnh >=224 px: 3.980 (98,56%).
- Ví dụ nhỏ nhất: `testing/Traffic Jam/2ed274cf-4d75-4057-94d1-3f1fc4b73f0c.jpg` (111×119). Lớn nhất: `training/Traffic Jam/150277323.jpg` (4.256×2.832).
- Rộng bất thường nhất: `training/Traffic Jam/images50.jpg` (387×130, 2,977); cao bất thường nhất: `training/High/151-not_low_volume.jpg` (250×350, 0,714).

## 6. Trùng lặp, rò rỉ và tính toàn vẹn split

SHA-256 theo byte phát hiện 142 nhóm trùng lặp (307 tệp); 105 nhóm đi qua ít nhất hai split. Số cặp cross-split: testing–training 51, testing–validation 24, training–validation 45 (tổng 120). Có **5 cặp theo tổ hợp** mang nhãn khác nhau, thuộc hai nhóm ảnh: một ảnh nằm ở `testing/High/...0285b1e6...jpg` và `training/Medium/...0285b1e6...jpg`; ảnh còn lại ở `testing/Low/...49e138c8...jpg` và `training/Medium/...49e138c8...jpg`.

Kiểm tra gần-trùng dùng dHash 64-bit trên grayscale 9×8, chỉ xét khác split, ngưỡng khoảng cách Hamming <=2 và loại byte-trùng: có 895 cặp ứng viên (testing–training 322; testing–validation 53; training–validation 520), 277 cặp khác nhãn. Đây là phép sàng lọc nhạy, không phải kết luận mọi cặp là cùng ảnh: cần xem thủ công hoặc xác nhận bằng SSIM/embedding trước khi gộp/xóa. Dù vậy, số byte-trùng đã đủ chứng minh rò rỉ nghiêm trọng.

Ví dụ trực quan trong contact sheet cũng cho thấy cùng cảnh/ảnh xuất hiện ở các split khác nhau. Do đó không dùng test hiện tại để báo cáo năng lực tổng quát; không chọn hyperparameter trên test.

## 7. Nhãn

| Thư mục hiện tại | Nhãn chuẩn đề xuất | Ghi chú |
|---|---|---|
| Empty | `empty` | Mật độ trống/rất thấp; nghĩa trực tiếp. |
| Low | `low` | Mật độ thấp. |
| Medium | `medium` | Mật độ trung bình. |
| High | `high` | Mật độ cao. |
| Traffic Jam | `traffic_jam` | Ùn tắc; thay khoảng trắng bằng gạch dưới chỉ ở mã/metadata. |

Tên thư mục hiện tại viết hoa nhất quán trong cả ba split; không nên đổi tên tệp/thư mục. Ánh xạ chuẩn chỉ nên tồn tại trong cấu hình/data loader. Hai ảnh byte-trùng nhưng nhãn mâu thuẫn cần adjudication nhãn trước khi dùng.

## 8. Tiền xử lý và chuẩn hóa đề xuất

**Độ phân giải:** 224×224 là lựa chọn hợp lý. Nó tương thích trực tiếp MobileNetV2/ResNet18 ImageNet, chi phí thấp, và chỉ 1,44% ảnh có cạnh dưới 224 nên đa số không bị phóng đại mạnh. Việc tăng lên 256/320 chưa có bằng chứng đặc thù dữ liệu để bù chi phí; có thể là thí nghiệm sau khi sửa split.

**Resize:** decode -> RGB -> resize giữ tỷ lệ sao cho vừa trong 224×224 -> letterbox/pad về 224×224 -> normalize. Đây là phương án đề xuất vì dữ liệu lẫn chủ yếu 4:3 và 16:9, cùng một số ảnh rất rộng/cao; kéo giãn làm biến dạng xe/làn đường, crop có thể cắt mất mặt đường và mật độ. Pad dùng màu trung tính hoặc reflection nhất quán; cần tránh để mô hình học màu viền nếu tỷ lệ khung tương quan với lớp. Direct stretch chỉ là baseline; resize+crop không nên là mặc định.

Dùng ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` nhất quán cho cả ba mô hình là lựa chọn thực dụng và công bằng trong so sánh, đặc biệt cho transfer learning. CNN từ đầu có thể hưởng lợi nhẹ từ thống kê riêng, nhưng nếu thử phải tính **chỉ từ training sau khi chốt lại split sạch** và dùng cùng thống kê cho validation/test; không tính từ toàn bộ dữ liệu hiện tại.

## 9. Đề xuất ablation augmentation

Các ảnh mẫu gồm camera đường bộ ban ngày, đêm, nhiều góc nhìn/camera và một phần ảnh Traffic Jam có phong cách nguồn khác. Không dùng vertical flip, xoay lớn, crop mạnh, hoặc biến đổi làm mất phần đường.

| Cấu hình | Biến đổi | Mục đích/ràng buộc |
|---|---|---|
| A0 — None | Chỉ resize-letterbox, RGB, normalize | Baseline tất định. |
| A1 — Mild | horizontal flip p=0,5; brightness/contrast ±10%; hue ±3° hoặc saturation ±10%; affine translate ≤3%, scale 0,95–1,05, rotate ±3° | Khác hướng camera, phơi sáng và căn khung nhẹ; không làm đổi mật độ. |
| A2 — Strong/environmental | A1 với brightness/contrast ±20%, gamma 0,8–1,2, saturation ±20%; Gaussian blur sigma 0–1,2 p≤0,25; haze/alpha trắng nhẹ 0–0,12 p≤0,20; rain nhẹ p≤0,15; shadow mờ p≤0,20; translate ≤5%, scale 0,90–1,10, rotate ±5° | Mô phỏng đêm/sáng, sương, mưa, rung/nhòe và chất lượng camera. Luôn kiểm tra mẫu trực quan để xe/làn đường vẫn thấy được. |

## 10. Kiểm tra trực quan

Đã xem contact sheet ngẫu nhiên có seed cố định, một ảnh cho mỗi tổ hợp split/lớp, cùng các ảnh có kích thước/tỷ lệ cực trị. Quan sát được: phần lớn ảnh là góc camera cao nhìn đường nhiều làn, thường có watermark/nhãn camera; có cả cảnh ban ngày, chạng vạng và đêm. Góc nhìn, độ rộng đường và mật độ xe khác nhau đáng kể. Mẫu Traffic Jam có cả cảnh ùn xe rõ rệt và phong cách ảnh/nguồn khác với phần camera giám sát; đây là rủi ro domain shift. Không thể xác nhận nhãn chỉ bằng 15 ảnh đại diện, đặc biệt khi kiểm tra byte-duplicate đã tìm thấy xung đột High/Medium và Low/Medium.

## 11. Rủi ro và bước tiếp theo

1. **Chặn huấn luyện/báo cáo điểm test** cho đến khi quyết định quy tắc làm sạch rò rỉ. Tạo split theo nhóm nội dung/nguồn (ít nhất nhóm SHA-256; tốt hơn nhóm near-duplicate/camera-sequence), không chỉ random theo ảnh.
2. Rà soát 105 nhóm byte-trùng xuyên split và hai nhóm xung đột nhãn; chỉ sau đó loại/giữ/cập nhật một bản ghi có chủ đích. Không tự động xóa dựa trên báo cáo này.
3. Xác thực thủ công các cặp dHash, ưu tiên 277 cặp khác nhãn và cặp liên quan test. `dataset_stats.json` có danh sách nhóm/cặp để truy vết.
4. Lớp Traffic Jam hiếm (5,62% train) và nguồn ảnh có vẻ dị thể hơn. Sau split sạch, dùng macro-F1/balanced accuracy cùng accuracy; khởi đầu bằng CrossEntropy có trọng số lớp. Focal Loss/WeightedRandomSampler chỉ nên là ablation sau baseline có weighted CE—không dùng đồng thời các cơ chế cân bằng nếu chưa chứng minh cần thiết.
5. Chốt quy tắc nhãn (đặc biệt ranh giới High/Traffic Jam), một seed/protocol split, augmentation và chỉ số chính trước khi chạy mô hình.

## Artefact kiểm toán

- `dataset_stats.json`: thống kê máy đọc được, toàn bộ nhóm trùng SHA-256 và 100 ví dụ near-duplicate đầu tiên.
- `dataset_contact_sheet_random.jpg`: 15 ảnh chọn ngẫu nhiên với seed cố định, một ảnh cho mỗi tổ hợp split/lớp; chỉ là bản sao render để kiểm tra, không sửa dữ liệu nguồn.
- `dataset_contact_sheet.jpg`: bản sheet đại diện theo thứ tự duyệt, dùng để đối chiếu nhanh.
