# Bài kiểm tra lần 2 — Nhóm 14 — Phan-loai-muc-do-benh-vong-mac-dai-thao-duong

- Fork: https://github.com/nnttrang05-design/hmyt-project
- Nhánh: `gd5-proposed-model`
- Thành viên và vai trò:
  - 20233777 - Huấn luyện mô hình & Đánh giá
  - 20233817 - Tiền xử lý & XAI (Grad-CAM)

## A. Literature Review và research gaps

### A1. Chiến lược tìm kiếm
| Nguồn | Từ khóa / truy vấn | Khoảng năm | Số kết quả | Số bài giữ lại |
|---|---|---|---|---|
| Google Scholar, PubMed, IEEE Xplore | "Diabetic Retinopathy", "BRSET", "Explainable AI", "Grad-CAM", "EfficientNet", "DenseNet" | 2021 - 2026 | ~150 | 15 |

**Tiêu chí chọn / loại bài:**
- **Chọn:** Bài báo Q1/Q2, sử dụng bộ dữ liệu BRSET hoặc các bộ dữ liệu lớn (APTOS, EyePACS), có tích hợp mô hình CNN (ResNet, DenseNet, EfficientNet) và có yếu tố XAI (Grad-CAM/Saliency map).
- **Loại:** Các bài báo không có yếu tố giải thích AI (XAI), bài báo chỉ dùng machine learning cổ điển, hoặc các bài báo có kết quả quá cao (98-99%) không rõ quy trình chống rò rỉ dữ liệu.

### A2. Ma trận y văn (15 bài tiêu biểu)
| # | Tên bài báo (Tác giả, Năm) | DOI | Dataset (n, nguồn) | Phương pháp | Kết quả chính | Kiểm định ngoài? | Hạn chế chính |
|---|---|---|---|---|---|---|---|
| 1 | A Lightweight Diabetic Retinopathy Detection Model Using a Deep-Learning Technique (2023) | `10.3390/diagnostics13193120` | APTOS, EyePacs | CLAHE + YOLO V7 + MobileNet V3-Small | APTOS Acc 98.0%, F1 93.7%; EyePacs Acc 98.4% | Không | Phụ thuộc chuyên gia; khó phát hiện tổn thương nhỏ; ảnh chất lượng thấp làm giảm độ chính xác. |
| 2 | Interpreting Deep Neural Networks in Diabetic Retinopathy Grading... (2025) | `10.3390/life15091473` | EyePACS, IDRiD, Messidor-2 | MobileNet + U-Net + Grad-CAM, IG | EyePACS Acc 82.9%; IDRiD Acc 49.5% | Không | Kém với DR-1/DR-2; không nhất quán ICDR; nhiễu do artifact; transferability kém. |
| 3 | ClairVue - Diabetic Retinopathy detection using EfficientNet and Grad-CAM (2025) | `10.2991/978-94-6463-858-5_82` | APTOS 2019 | RA-EfficientNet + Grad-CAM | Acc tổng 96.5%; AUC ~0.89-0.96 | Không | Dataset mất cân bằng; phân loại kém ở ranh giới; chưa kiểm chứng lâm sàng. |
| 4 | Multi-Scale Attention Network for Diabetic Retinopathy Classification (2021) | `10.1109/ACCESS.2021.3070685` | EyePACS, APTOS 2019 | MSA-Net (ResNetAS + attention) | APTOS Acc 84.6%, Kappa 0.896 | Không | Nhầm lẫn giữa các lớp tương quan cao; weighted loss làm giảm nhẹ hiệu suất. |
| 5 | A clinically interpretable deep learning pipeline using EfficientNet... and GradCAM (2026) | `10.1186/s12886-026-04897-4` | APTOS 2019 | EfficientNet-B1 + Mixup/CutMix + Focal Loss + Grad-CAM | Test Acc 84.97%, Macro F1 0.6627 | Không | Overfitting chưa loại bỏ; chỉ dùng 1 dataset; Severe DR recall thấp. |
| 6 | Privacy-aware diabetic retinopathy grading... mixture-of-experts federated... (2026) | `10.1038/s41598-026-58292-6` | EyePACS, APTOS-2019, IDRiD | FL-MoE (CNN-LSTM, ViT, ResNet152...) + Grad-CAM | EyePACS Acc 76.2%, QWK 36.8; APTOS Acc 71.0% | Không | Mô phỏng, chưa lâm sàng; Grad-CAM localization kém (IoU <0.03); QWK thấp. |
| 7 | Tri-attention-augmented residual CNN with XAI for early DR screening... (2026) | `10.1038/s41598-026-58716-3` | APTOS, IDRiD | ResNet-50 + Tri-Attention + Grad-CAM | Nội bộ Acc 97.01%; Ngoại bộ IDRiD Acc 99.4% | Có (IDRiD) | Gộp Severe + PDR; downsampling mất dữ liệu; Grad-CAM chưa định lượng. |
| 8 | A multi model deep net with an explainable AI based framework for DR... (2025) | `10.1038/s41598-025-93376-9` | DiaRetDB1, APTOS, EyePACS | Modified U-Net, DenseNet + Grad-CAM | APTOS Acc 98.98%; EyePACS Acc 99.12% | Không | Độ phức tạp cao; Grad-CAM chưa định lượng; kết quả 99% nghi ngờ overfitting. |
| 9 | Multi-Channel Feature Fusion and Monte Carlo Dropout for Uncertainty-Aware DR... (2026) | `10.48550/arXiv.2608.15234` | APTOS-2019 | EfficientNetV2-L + MC-dropout + Grad-CAM | QWK 91.31%; PDR recall 36.4% | Không | Data leakage; MC-dropout chưa calibration; Grad-CAM diffuse; miss PDR. |
| 10 | Retinal Disease Classification Using Deep CNN on Fundus Images (2025) | `10.56313/jictas.v4i2.451` | APTOS 2019 | ResNet50 + Grad-CAM | Acc 92.4%, F1 0.91, Kappa 0.89 | Không | Chỉ dùng APTOS; mất cân bằng lớp; Grad-CAM chưa định lượng; nhầm lẫn Moderate/Severe. |
| 11 | VG-RETFound: a hybrid retinal AI framework... (2026) | `10.3389/fmed.2026.1905129` | RFMiD, ODIR-5K, BRSET | RETFound + Graph Attention | Nội bộ Acc 96.39%; Ngoại bộ BRSET Acc 94.81% | Có (BRSET) | Cần kiểm chứng đa trung tâm; chưa có đánh giá XAI chi tiết. |
| 12 | Deep Learning for Comprehensive Analysis of Retinal Fundus Images (2025) | `10.3390/bioengineering12080840` | BRSET | Swin Transformer | DR (2 lớp) AUC 0.98; DR (3 lớp) F1 0.95 | Không | Chưa đánh giá tổng quát hóa ngoài BRSET; hạn chế diễn giải lâm sàng sâu. |
| 13 | BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos (2024) | `10.1371/journal.pdig.0000454` | BRSET | Bài giới thiệu dataset | 16.266 ảnh, 8.524 BN | Không | Dataset chỉ từ São Paulo; chưa đại diện toàn bộ dân số. |
| 14 | RetinoMamba-AnatomyQuery: Anatomy-Guided Query Learning... (2026) | `10.3390/life16091577` | ODIR-2019, BRSET, RFMiD | Vision Mamba + EfficientNet-B3 | Nội bộ Acc 97.93%; Ngoại bộ RFMiD Acc 96.21% | Có (RFMiD) | Chỉ phân loại nhị phân; chưa phân loại đa bệnh/phân độ DR. |
| 15 | A clinically interpretable deep learning framework using EfficientNetV2... and GradCAM (2026) | `10.1186/s12886-026-04897-4` | APTOS 2019 | EfficientNet-B1 + Focal Loss + TTA + Grad-CAM | Test Acc 83.61%, AUC 0.9966, Macro F1 0.6027 | Không | Overtuning khi gộp lớp; 1 dataset (APTOS); cần external validation. |

### A3. Research gaps
| Mã | Khoảng trống | Bằng chứng (số # trong A2) | Dự án của nhóm giải quyết bằng cách nào |
|---|---|---|---|
| G1 | Thiếu sự kết hợp giữa kiến trúc CNN nhẹ và XAI trên bài toán 5 lớp ICDR trên BRSET. | Bài #12, #14 (dùng Transformer/Mamba nặng); Bài #13 (chưa có DL) | Sử dụng ResNet-18, DenseNet-121 kết hợp Grad-CAM để giải thích quyết định trên 5 lớp. |
| G2 | Chưa giải quyết triệt để vấn đề mất cân bằng dữ liệu cực đoan. | Bài #5 (Severe DR recall thấp), Bài #2 (kém với DR-1/2) | Sử dụng Weighted Cross-Entropy Loss, báo cáo F1, Sensitivity, FP/FN thay vì chỉ Accuracy. |
| G3 | Thiếu kiểm soát rủi ro thiên lệch và rò rỉ dữ liệu (Data Leakage). | Bài #8 (kết quả 99% nghi ngờ overfitting/leakage), Bài #9 (Data leakage) | Áp dụng nghiêm ngặt Patient-level Split, chứng minh 0% rò rỉ bằng code. |

### A4. Thẩm định 1 bài quan trọng nhất trong A2 bằng PROBAST+AI
**Bài số #2:** Interpreting Deep Neural Networks in Diabetic Retinopathy Grading: A Comparison with Human Decision Criteria (2025).
**Lý do chọn:** Đây là bài báo đi sâu vào đánh giá XAI (Grad-CAM, IG) và so sánh với tiêu chuẩn con người (ICDR), rất sát với mục tiêu cốt lõi của nhóm.

| Miền | Nguy cơ sai lệch | Căn cứ (Trích dẫn chính xác từ bài báo) |
|---|---|---|
| **Người tham gia và nguồn dữ liệu** | **Cao** | Mục 5.2 (Datasets) và Bảng 5: Sử dụng nhiều tập dữ liệu (EyePACS, IDRiD, Messidor-2). Kết quả transferability rất kém khi test chéo (mô hình train trên EyePACS nhưng test trên IDRiD chỉ đạt Accuracy 0.4951). |
| **Biến dự báo** | **Thấp** | Mục 5.3 (Training Setup) và Mục 5.6 (Interpretation): Sử dụng ảnh fundus gốc, resize về 672x672, chuẩn hóa pixel [-1, 1] rõ ràng. Sử dụng mạng MobileNet pretrained ImageNet. |
| **Kết cục** | **Trung bình** | Mục 6.1 (Results): Tác giả chỉ ra sự mất cân bằng lớp (class imbalance) trầm trọng khiến mô hình bị bias về DR-0. Mục 7.1 cũng thừa nhận ranh giới giữa các mức độ DR-1, DR-2 rất mờ nhạt, ngay cả với bác sĩ nhãn khoa (HRPs). |
| **Phân tích** | **Cao** | Mục 7.1 (Manual Analysis) và Mục 7.2 (Automatic Analysis): XAI (Grad-CAM) tạo ra bản đồ nhiệt thô, bị nhiễu do các vết bẩn trên ống kính máy chụp (Hình 5). Việc đánh giá định lượng XAI bằng segmentation (Hình 6) cho kết quả IoU rất thấp (chỉ 0.0052 và 0.0188). |

**Tính áp dụng cho bài toán của nhóm:**
Bài báo này là lời cảnh báo rằng việc dùng XAI chưa đủ, mà XAI còn phải nhất quán và không bị đánh lừa bởi nhiễu thiết bị. Nhóm đã học được cách đánh giá Grad-CAM khắt khe hơn (dùng ClassifierOutputTarget để giải thích đúng nhãn thực tế), đồng thời nhận thức được tầm quan trọng của việc tiền xử lý (Auto-Crop ROI, CLAHE) để triệt tiêu các vết bẩn/viền đen trước khi đưa vào mô hình.

---

## B. Kết quả và timeline

### B1. Kết quả hiện có
| Mô hình | Tập đánh giá | Metric chính | Metric phụ | Nguồn số liệu |
|---|---|---|---|---|
| Baseline 1 (ResNet-18) | Test (Patient-level, n=1619) | Acc 93.08%, F1-Macro 53.86% | Sens (Nhãn 4): 80.77%, Spec: 98.28% | Kaggle Notebook (Cell 9 - Classification Report) |
| Mô hình đề xuất (DenseNet-121) | Test (Patient-level, n=1619) | Acc 94.19% | FN (Nhãn 4) chỉ 7 ca | Kaggle Notebook (Cell 9 - Classification Report) |
| Đang chạy (EfficientNet-B0) | Chưa có | Chưa có | Chưa  | Đang huấn luyện |

### B2. Ablation (Mốc 3)
| Cấu hình | Metric chính | Chênh lệch | Nguồn |
|---|---|---|---|
| Đầy đủ (ROI + CLAHE + Weighted Loss) | Acc 93.08% | Không áp dụng | Cell 8-9 |
| Chưa thực hiện (Dự kiến: Bỏ CLAHE, Bỏ Circular Masking) | Chưa có | Không áp dụng | Sẽ làm ở Mốc 4 |

### B3. Đối chiếu với y văn
So sánh kết quả B1 với bài #2 (Interpreting DNNs, 2025):
- **Điểm giống:** Cùng sử dụng các mô hình CNN (MobileNet/ResNet) và phương pháp XAI.
- **Điểm không công bằng:** Bài #2 đánh giá trên nhiều bộ dữ liệu (đa nguồn) nên kết quả transferability thấp (49.5% trên IDRiD). Nhóm chỉ đánh giá trên tập Test của BRSET, do đó kết quả cao hơn (94.19% với DenseNet) nhưng chưa chứng minh được khả năng tổng quát hóa.

### B4. Timeline đến khi hoàn thành
| Việc | Phụ trách | Mốc | Ngày dự kiến | Trạng thái |
|---|---|---|---|---|
| Huấn luyện ResNet-18 | Khánh Ly | M3 | 07/11 | Xong (Acc 93.08%) |
| Huấn luyện DenseNet-121 | Khánh Ly | M3 | 08/11 | Xong (Acc 94.19%) |
| Huấn luyện EfficientNet-B0 | Thùy Trang | M3 | 09/11 | Đang chạy |
| Vẽ biểu đồ XAI (Grad-CAM) | Thùy Trang | M4 | 10/11 | Xong (ResNet, DenseNet) |
| Hoàn thiện báo cáo & Repo | Cả 2 | Final | 15/11 | Đang làm |

### B5. Rủi ro / đang vướng / câu hỏi cho GV
- **Rủi ro:** Lớp 3 (Nặng) có quá ít dữ liệu (78 ảnh), dẫn đến Macro F1-Score rất thấp dù dùng Weighted Loss.
- **Câu hỏi:** Thầy có thể gợi ý phương pháp nào để cải thiện hiệu năng cho lớp thiểu số này ngoài việc dùng Class Weights không ạ? (Ví dụ: dùng thêm Data Augmentation chuyên biệt, hay chấp nhận nó là giới hạn của dữ liệu?)

---

## C. Tự đánh giá dự án theo Phụ lục FAIR, TRIPOD+AI, PROBAST+AI

### C1. FAIR — dữ liệu
| Mục | Trả lời | Bằng chứng |
|---|---|---|
| Nguồn, phiên bản, ngày tải | BRSET 1.0.1, tải từ Kaggle | File `README.md`, script tự động tìm đường dẫn trong thư mục `src/` |
| Giấy phép / DUA | Công khai, tuân thủ giấy phép phi thương mại | Không chia sẻ lại ảnh trong repo (đã cấu hình `.gitignore`) |
| Datasheet | 16.266 ảnh, 8.524 BN, 102 cột, 5 nhãn ICDR | Notebook EDA (Cell 1, 2) |
| Không có định danh bệnh nhân | Đã ẩn danh bằng `patient_id` | Code kiểm tra `patient_id` (0% trùng lặp) |

### C2. Hướng dẫn báo cáo (TRIPOD+AI)
| Chủ đề báo cáo | Những gì nhóm đã thực hiện và báo cáo rõ |
| :--- | :--- |
| *Nguồn dữ liệu & Người tham gia* | BRSET 1.0.1 (16.266 ảnh, 8.524 BN). Chia tập Patient-level Split (Train: 13.015, Val: 1.632, Test: 1.619). Đã dùng code `assert` để chứng minh 0% bệnh nhân trùng lặp giữa 3 tập. |
| *Thời điểm dự đoán (T0)* | Thời điểm chụp ảnh đáy mắt tại bệnh viện. Mô hình chỉ sử dụng duy nhất 1 ảnh tại T0, không dùng thông tin tương lai (như thời gian mắc bệnh) để tránh rò rỉ dữ liệu. |
| *Biến dự báo (Input)* | Ảnh fundus trải qua pipeline: Auto-Crop ROI → Circular Masking → CLAHE (không gian màu LAB) → Resize 384x384 → Normalize (mean/std ImageNet). |
| *Mô hình & Huấn luyện* | Sử dụng ResNet-18, DenseNet-121 (đã xong) và EfficientNet-B0 (đang chạy). Optimizer: AdamW (LR=1e-4). Hàm mất mát: Weighted Cross-Entropy. Kỹ thuật: Mixed Precision (AMP), Early Stopping (Patience=5), 30 Epochs. |
| *Đánh giá hiệu năng* | Báo cáo chi tiết: Accuracy, Macro F1-Score, Sensitivity, Specificity, Confusion Matrix và biểu đồ phân tích lỗi (FP/FN). Không chỉ báo cáo mỗi Accuracy. |
| *Giải thích AI (XAI)* | Sử dụng Grad-CAM (Target layer: `layer4[-1]` cho ResNet, `features[-1]` cho DenseNet/EfficientNet). Sử dụng `ClassifierOutputTarget` để giải thích đúng nhãn thực tế. |

### C3. PROBAST+AI — tự thẩm định dự án của nhóm
| Miền đánh giá | Nguy cơ sai lệch | Căn cứ trong repo | Việc sẽ làm để giảm nguy cơ |
| :--- | :--- | :--- | :--- |
| *Người tham gia & Nguồn* | **Trung bình** | Chỉ dùng BRSET (Brazil), chưa có external validation. | Ghi nhận vào phần Hạn chế. |
| *Biến dự báo (Ảnh)* | **Thấp** | Đã Auto-Crop ROI, CLAHE, Masking. Chỉ augment trên Train. | Giữ nguyên pipeline. |
| *Kết cục* | **Trung bình** | Nhãn 3 quá hiếm, AI dự đoán kém. | Báo cáo trung thực, phân tích FN. |
| *Phân tích* | **Thấp** | Chia Patient-level, fit class weights chỉ trên Train, có Early Stopping. | Tiếp tục theo dõi. |

**Tự đánh giá chung:** Nhóm đánh giá nguy cơ thiên lệch của mô hình ở mức **Trung bình - Thấp**. Các nguy cơ lớn nhất về rò rỉ dữ liệu và mất cân bằng lớp đã được xử lý triệt để bằng Patient-level split và Weighted Loss. Hạn chế lớn nhất còn lại là **chưa có External Validation**, đây sẽ là hướng phát triển bắt buộc trong tương lai.

### C4. Bảng kiểm nhanh
| Mục | Đạt / Chưa | Bằng chứng |
|---|---|---|
| Datasheet | Đạt | Notebook EDA |
| Phiên bản + checksum | Đạt | Đường dẫn `/kaggle/input/...` |
| Không định danh | Đạt | `patient_id` |
| Chọn đúng hướng dẫn | Đạt | TRIPOD+AI |
| Định nghĩa kết cục, nhãn, T0 | Đạt | 5 nhãn DR_ICDR, T0 là lúc chụp ảnh |
| Chia theo bệnh nhân | Đạt | Code Assert 0% leak |
| Mô hình mốc | Đạt | ResNet-18 |
| AUC + calibration | **Chưa** | Chưa vẽ Calibration curve (sẽ làm nếu kịp) |
| Cỡ mẫu đủ | Đạt | 13.015 ảnh Train |

---

## D. Khai báo dùng AI
| Công cụ | Dùng cho phần nào | Nhóm đã kiểm chứng lại bằng cách nào |
|---|---|---|
| ChatGPT / Gemini | Gợi ý cấu trúc code module hóa, sửa lỗi logic code (Data Leakage, Scheduler), gợi ý cách phân tích biểu đồ FP/FN. | Đã tự chạy lại code trên Kaggle, kiểm tra output và số liệu thực tế khớp với báo cáo. |
| OpenCode (VSCode) | Tạo cấu trúc thư mục và viết docstring cho các file `.py`. | Đã review lại code, đảm bảo import đúng và chạy được logic. |

---

## TÀI LIỆU THAM KHẢO (15 bài báo trong A2)
1. A Lightweight Diabetic Retinopathy Detection Model Using a Deep-Learning Technique (2023). *Diagnostics*, 13(19), 3120. DOI: `10.3390/diagnostics13193120`
2. Interpreting Deep Neural Networks in Diabetic Retinopathy Grading: A Comparison with Human Decision Criteria (2025). *Life*, 15(9), 1473. DOI: `10.3390/life15091473`
3. ClairVue - Diabetic Retinopathy detection using EfficientNet and Grad-CAM (2025). DOI: `10.2991/978-94-6463-858-5_82`
4. Multi-Scale Attention Network for Diabetic Retinopathy Classification (2021). *IEEE Access*. DOI: `10.1109/ACCESS.2021.3070685`
5. A clinically interpretable deep learning pipeline for diabetic retinopathy classification using EfficientNet, advanced data augmentation and GradCAM (2026). *BMC Ophthalmology*. DOI: `10.1186/s12886-026-04897-4`
6. Privacy-aware diabetic retinopathy grading and visual lesion-focused interpretability through mixture-of-experts federated deep learning with explainable AI (2026). *Scientific Reports*. DOI: `10.1038/s41598-026-58292-6`
7. Tri-attention-augmented residual convolutional neural network with explainable AI for automated early diabetic retinopathy screening in resource-constrained environments (2026). *Scientific Reports*. DOI: `10.1038/s41598-026-58716-3`
8. A multi model deep net with an explainable AI based framework for diabetic retinopathy segmentation and classification (2025). *Scientific Reports*. DOI: `10.1038/s41598-025-93376-9`
9. Multi-Channel Feature Fusion and Monte Carlo Dropout for Uncertainty-Aware Diabetic Retinopathy Grading (2026). *arXiv*. DOI: `10.48550/arXiv.2608.15234`
10. Retinal Disease Classification Using Deep CNN on Fundus Images (2025). *JICTAS*. DOI: `10.56313/jictas.v4i2.451`
11. VG-RETFound: a hybrid retinal AI framework for early vision impairment screening through vessel-graph learning and cross-modal fusion (2026). *Frontiers in Medicine*. DOI: `10.3389/fmed.2026.1905129`
12. Deep Learning for Comprehensive Analysis of Retinal Fundus Images: Detection of Systemic and Ocular Conditions (2025). *Bioengineering*, 12(8), 840. DOI: `10.3390/bioengineering12080840`
13. Nakayama, L. F., et al. (2024). BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos. *PLOS Digital Health*, 3(7), e0000454. DOI: `10.1371/journal.pdig.0000454`
14. RetinoMamba-AnatomyQuery: Anatomy-Guided Query Learning for Efficient Vision Impairment Screening from Retinal Fundus Images (2026). *Life*, 16(9), 1577. DOI: `10.3390/life16091577`
15. A clinically interpretable deep learning framework for diabetic retinopathy classification using EfficientNetV2, advanced data augmentation and GradCAM (2026). *BMC Ophthalmology*. DOI: `10.1186/s12886-026-04897-4`
