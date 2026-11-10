# PHÁT HIỆN TRẦM CẢM Ở SINH VIÊN
### Student Depression Detection – Data Analysis & Machine Learning

> **Cảnh báo:** Đây là dự án học tập về phân tích dữ liệu và phân loại nhãn trong một bộ dữ liệu. Mô hình không phải công cụ chẩn đoán, không xác định một cá nhân có mắc trầm cảm hay không và không thay thế chuyên gia sức khỏe tâm thần.

## 1. Giới thiệu

Dự án sử dụng bộ dữ liệu `Student_Depression_Dataset.csv` để:
- Khám phá cấu trúc và chất lượng dữ liệu.
- Phân tích phân bố nhãn `Depression` và các thuộc tính trong tập dữ liệu.
- Trực quan hóa dữ liệu và quan sát mối liên hệ thống kê.
- Tiền xử lý dữ liệu và xây dựng mô hình Machine Learning phân loại nhị phân.
- Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score, Confusion Matrix và ROC-AUC.
- Thảo luận giới hạn, nguy cơ thiên lệch và khả năng khái quát.

**Bài toán:** Supervised Binary Classification.  
**Biến mục tiêu:** `Depression` với hai nhãn `Yes` và `No` trong file hiện tại.

## 2. Thông tin bộ dữ liệu

File được kiểm tra ban đầu có **4.464 dòng và 13 cột**, không có giá trị thiếu và không có dòng trùng lặp hoàn toàn tại thời điểm kiểm tra. Notebook sẽ đọc lại và xác minh các con số mỗi lần chạy.

| Cột | Mô tả sơ bộ |
|---|---|
| `Gender` | Nhóm giới tính được ghi trong dữ liệu |
| `Age` | Tuổi |
| `Family_background` | Nhóm hoàn cảnh gia đình |
| `Relationship_status` | Trạng thái mối quan hệ |
| `Marital_status` | Tình trạng hôn nhân |
| `Living_style` | Hình thức sinh sống |
| `Current_semester` | Học kỳ hiện tại |
| `CGPA` | Điểm trung bình tích lũy; cần xác minh thang điểm |
| `friend_circle` | Thuộc tính Yes/No liên quan bạn bè; cần xác minh câu hỏi gốc |
| `Health_condition` | Nhóm tình trạng sức khỏe theo nhãn trong dữ liệu; cần xác minh định nghĩa |
| `Cost_without_study` | Tên trường chi phí chưa rõ định nghĩa; không tự diễn giải khi chưa có tài liệu gốc |
| `Home_division` | Khu vực/đơn vị địa lý trong bộ dữ liệu |
| `Depression` | Nhãn mục tiêu Yes/No |

### TODO – cần xác minh
1. Nguồn dữ liệu và URL phát hành chính thức.
2. Tác giả/đơn vị thu thập, thời điểm và phương pháp lấy mẫu.
3. Định nghĩa câu hỏi và cách mã hóa của từng cột.
4. Tiêu chí gán nhãn `Depression`: tự khai báo, thang đo khảo sát hay nguồn nào khác?
5. Giấy phép/quyền sử dụng dữ liệu, đặc biệt trước khi tải lên GitHub.
6. Ý nghĩa chính xác của `Cost_without_study`, `friend_circle`, `Health_condition`.

Không được mô tả nhãn `Depression` là chẩn đoán lâm sàng nếu nguồn không xác nhận điều đó.

## 3. Cấu trúc thư mục đề xuất

```text
BaoCaoCuoiKy/
├── README.md
├── PhatHienTramCamSinhVien.ipynb
├── requirements.txt
├── .gitignore
└── data/
    └── Student_Depression_Dataset.csv
```

Nếu notebook nằm trong `notebooks/`, cập nhật đường dẫn dữ liệu trong phần đọc CSV hoặc dùng một trong các đường dẫn mà notebook đã kiểm tra.

**Bảo mật dữ liệu:** Không đẩy dữ liệu lên repository công khai nếu giấy phép không cho phép, hoặc nếu file chứa thông tin cá nhân/nhạy cảm. Nếu không thể chia sẻ CSV, README nên hướng dẫn tải dữ liệu từ nguồn được phép.

## 4. Cài đặt môi trường

Khuyến nghị Python 3.10 trở lên.

### Windows – Git Bash/Terminal

```bash
python -m venv .venv
```

PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Git Bash:
```bash
source .venv/Scripts/activate
```

Cài thư viện:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu chưa có `requirements.txt`, có thể tạo nội dung ban đầu như sau:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
ipykernel
joblib
```

## 5. Cách chạy Notebook

1. Mở thư mục dự án trong VS Code hoặc Jupyter.
2. Đặt CSV vào `data/Student_Depression_Dataset.csv` hoặc cùng thư mục với notebook.
3. Cài các thư viện trong `requirements.txt`.
4. Mở `PhatHienTramCamSinhVien.ipynb`.
5. Chọn kernel Python thuộc môi trường `.venv`.
6. Chạy lần lượt từng cell hoặc chọn **Restart Kernel and Run All**.
7. Đọc các nhận xét và hoàn thiện những vị trí `TODO`.

Nếu gặp `FileNotFoundError`, kiểm tra tên file, phần mở rộng `.csv`, chữ hoa/thường và biến `DATA_PATH` ở mục 3 của notebook.

## 6. Quy trình thực hiện trong Notebook

| Giai đoạn | Nội dung |
|---|---|
| 1. Khởi tạo | Giới thiệu đề tài, mục tiêu, import thư viện |
| 2. Đọc dữ liệu | Đọc CSV, xem kích thước, 5 dòng đầu |
| 3. Khảo sát dữ liệu | Kiểu dữ liệu, thống kê mô tả, giá trị duy nhất |
| 4. Làm sạch | Kiểm tra thiếu, trùng, khoảng giá trị, chuẩn hóa khoảng trắng |
| 5. EDA | Phân bố nhãn, biểu đồ biến số/biến phân loại, so sánh theo nhãn |
| 6. Tiền xử lý | Tách X/y, chia train/test, điền thiếu, scale và one-hot |
| 7. Huấn luyện | Dummy Classifier, Logistic Regression, Random Forest |
| 8. Đánh giá | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, cross-validation |
| 9. Phân tích | Permutation importance và nhận xét đặc trưng |
| 10. Kết luận | Thảo luận kết quả, hạn chế, hướng phát triển |

## 7. Các mô hình và chỉ số đánh giá

- **DummyClassifier:** baseline để so sánh.
- **Logistic Regression:** mô hình phân loại tuyến tính.
- **Random Forest:** mô hình tổ hợp nhiều cây quyết định.

Các chỉ số:
- **Accuracy:** tỷ lệ dự đoán đúng tổng thể.
- **Precision (Yes):** tỷ lệ đúng trong các dự đoán Yes.
- **Recall (Yes):** tỷ lệ mẫu Yes được mô hình nhận diện.
- **F1-score (Yes):** cân bằng Precision và Recall.
- **ROC-AUC:** khả năng phân biệt hai lớp theo các ngưỡng xác suất.
- **Confusion Matrix:** thể hiện số dự đoán đúng/sai theo từng lớp.

Không chọn mô hình chỉ dựa trên Accuracy. Cần so sánh Recall, Precision, F1, ROC-AUC và kết quả cross-validation. Các chỉ số chỉ phản ánh khả năng khớp với nhãn của dataset này.

## 8. Những điểm cần xem xét khi phân tích

- Xác minh cách thu thập dữ liệu và tiêu chí gán nhãn.
- Kiểm tra nguy cơ data leakage, đặc biệt các biến có thể được đo từ hoặc sau kết quả khảo sát.
- Không suy diễn quan hệ nhân quả từ biểu đồ hoặc tương quan.
- Xem xét dữ liệu mất cân bằng lớp.
- Đánh giá sự khác biệt giữa các nhóm và nguy cơ thiên lệch nếu có cơ sở.
- Không đưa ra khuyến nghị y tế cá nhân từ kết quả mô hình.

## 9. Kết quả cần bổ sung sau khi chạy

**TODO – cập nhật bằng số liệu notebook thực tế:**

| Nội dung | Kết quả |
|---|---|
| Số dòng sau làm sạch | `[điền]` |
| Tỷ lệ nhãn Yes | `[điền %]` |
| Tỷ lệ nhãn No | `[điền %]` |
| Mô hình được chọn | `[điền]` |
| Accuracy | `[điền]` |
| Precision – Yes | `[điền]` |
| Recall – Yes | `[điền]` |
| F1-score – Yes | `[điền]` |
| ROC-AUC | `[điền]` |
| Đặc trưng nổi bật | `[điền sau khi xem permutation importance]` |

### TODO – viết kết luận
Nêu mô hình nào tốt hơn baseline, chỉ số nào là căn cứ lựa chọn, mô hình có ổn định qua cross-validation không, và những giới hạn của bộ dữ liệu. Không khẳng định mô hình có thể chẩn đoán trầm cảm.

## 10. Nguồn dữ liệu và tài liệu tham khảo

- **Dataset:** `Student_Depression_Dataset.csv`
- **URL nguồn:** `[TODO – điền đường dẫn nguồn phát hành]`
- **Tác giả/đơn vị:** `[TODO]`
- **Ngày truy cập:** `[TODO]`
- **Giấy phép:** `[TODO – xác minh quyền sử dụng]`
- **Tài liệu mô tả biến:** `[TODO – thêm link nếu có]`

## 11. Checklist trước khi nộp

- [ ] Xác minh nguồn và giấy phép dataset.
- [ ] Xác minh ý nghĩa 13 cột.
- [ ] Xác minh cách tạo nhãn `Depression`.
- [ ] Đọc và chạy toàn bộ notebook không lỗi.
- [ ] Kiểm tra data leakage và lựa chọn đặc trưng.
- [ ] Điền nhận xét cho các biểu đồ.
- [ ] Điền bảng kết quả mô hình.
- [ ] Viết kết luận dựa trên kết quả thực tế.
- [ ] Kiểm tra dữ liệu có được phép đẩy lên GitHub không.
- [ ] Xóa dữ liệu nhạy cảm/không được phép công khai khỏi repository.

---
**Lưu ý học thuật:** Dự án là minh họa kỹ thuật về phân tích dữ liệu và Machine Learning. Mọi kết quả chỉ nên được diễn giải trong phạm vi bộ dữ liệu, không dùng để chẩn đoán hoặc ra quyết định chăm sóc sức khỏe cá nhân.