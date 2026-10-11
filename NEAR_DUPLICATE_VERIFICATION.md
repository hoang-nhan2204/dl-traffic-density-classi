# Xác minh near-duplicate xuyên split

## Phạm vi và phương pháp

Xác minh này chỉ đọc `traffic.zip`; không thay đổi ảnh, nhãn, split hay manifest. Các split ghi dưới đây là split gốc trong đường dẫn ZIP (`training`, `validation`, `testing`), vì manifest active hiện không có trong working tree.

Tái tạo candidate theo dHash grayscale 64-bit, chỉ cặp khác split, Hamming <=2 và loại byte-identical SHA-256. Kết quả là **895 cặp**, khớp số lượng audit trước. Với mỗi cặp, tính thêm:

- pHash DCT 8×8 grayscale; báo cáo Hamming distance.
- Global SSIM grayscale sau resize 224×224.
- Cosine similarity embedding từ MobileNetV2 ImageNet pretrained (feature vector trước classifier); không dùng model huấn luyện trên dataset này.
- Width/height, aspect ratio và format được lấy trong lúc xử lý; CSV pair tập trung các cột bắt buộc, còn ảnh nguồn giữ nguyên trong ZIP.

Phân loại dùng đồng thời các tín hiệu và contact sheet. A rất nghiêm ngặt. B rộng hơn có chủ đích: “có khả năng cùng cảnh/camera hoặc frame kế cận”, **không** đồng nghĩa tự động group. Đây là cách bảo thủ vì traffic camera có nhiều cảnh tĩnh làm embedding/hash cao dù không phải cùng frame.

## Kết quả định lượng

| Hạng mục | Số lượng |
|---|---:|
| Candidate dHash cross-split (không exact duplicate) | 895 cặp |
| Ảnh unique tham gia candidate | 604 |
| A — confirmed near-duplicate | 18 cặp; 28 ảnh; 11 connected groups |
| B — likely related / same sequence | 794 cặp |
| C — false positive / không đủ bằng chứng liên quan | 83 cặp |
| A+B | 812 cặp; 561 ảnh; 95 connected groups |
| Tất cả candidate khác nhãn | 277 cặp |
| A khác nhãn | 0 cặp |
| A+B khác nhãn | 233 cặp, thuộc 26 group |

**Ảnh hưởng đến split gốc:** A có 11/11 group đi qua split (15 training, 9 validation, 4 testing ảnh), nên đều là leakage chắc chắn. Nếu group A+B, 95/95 group đi qua split (381 training, 103 validation, 77 testing ảnh); quy mô lớn và 26 group chứa xung đột nhãn nên không phù hợp để tự động áp dụng.

## Diễn giải category

- **A:** pHash rất thấp (0–2), SSIM 0,989–0,999, cosine 0,980–0,991; contact sheet cho thấy cùng frame/cảnh gần như trùng hoàn toàn, chỉ khác encode hoặc thay đổi rất nhỏ. Đây là leakage train/val/test rõ ràng.
- **B:** nhiều cặp cùng góc camera, thời điểm/cảnh rất gần hoặc traffic thay đổi nhẹ. Signal trung vị: pHash 2, SSIM 0,983, cosine 0,953 trước khi mở rộng B; nhóm B mở rộng bao gồm các cặp được xem trực quan có cùng scene nhưng embedding thấp hơn. Một số cặp khác nhãn là dấu hiệu frame liền nhau có thể nằm ở ngưỡng density, hoặc label không nhất quán.
- **C:** không đủ sự đồng thuận từ pHash, SSIM và embedding để gọi là cùng scene/frame một cách bảo thủ. Không dùng C để group.

## Kiểm tra trực quan

Contact sheets nằm tại `outputs/near_duplicate_audit/`:

- `top100_*.jpg`: 100 cặp similarity cao nhất.
- `conflicts_*.jpg`: toàn bộ 277 cặp khác nhãn.
- `boundary50_*.jpg`: 50 cặp gần vùng quyết định.
- `category_A_*.jpg`, `category_B_*.jpg`, `category_C_*.jpg`: ví dụ từng category.

Mỗi hàng hiển thị 2 ảnh, split/label, dHash, pHash, SSIM, cosine và tên file. Quan sát top-A xác nhận same-frame/cùng cảnh rất rõ. Contact sheet khác nhãn cho thấy nhiều B là cùng camera với xe thay đổi nhẹ; do đó không nên coi mọi B là duplicate để xóa hay group tự động.

## Quy tắc grouping đề xuất

**Khuyến nghị: group only A.**

Áp dụng automatic grouping chỉ khi ba tín hiệu mạnh cùng đồng ý, theo rule đã dùng cho A:

```text
(pHash <= 6 AND SSIM >= 0.90 AND cosine >= 0.980)
OR
(pHash <= 3 AND SSIM >= 0.84 AND cosine >= 0.985)
```

Các A cùng label cần chung `group_id` khi tạo split, không tự động xóa. Hiện không có A cross-label. Với B, chỉ review thủ công các group xuyên split, ưu tiên 233 cặp khác nhãn và các B bao gồm test; nếu xác nhận là cùng chuỗi/frame gần nhau thì group, nếu nhãn thực sự mâu thuẫn thì quyết định adjudicate/exclude ở bước riêng. Không group C.

Không có một ngưỡng tự động đáng tin cậy để phân biệt toàn bộ B khỏi cảnh camera tĩnh nhưng độc lập; hybrid rule A-auto + B-manual là phù hợp hơn group A+B.

## Artefact machine-readable

- `outputs/near_duplicate_audit/near_duplicate_pairs.csv`: 895 rows, cột `image_a`, `image_b`, split/label hai phía, dHash, pHash, SSIM, cosine, category, notes.
- `outputs/near_duplicate_audit/near_duplicate_groups.csv`: connected group cho scope A-only và A-plus-B.

Các artifact này là verification output; pipeline training và manifest chưa được sửa hay tái sinh.
