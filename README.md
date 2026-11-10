# PHÁT HIỆN TRẦM CẢM Ở SINH VIÊN
### Student Depression Detection – Data Analysis & Machine Learning

## 1. Giới thiệu

Dự án sử dụng bộ dữ liệu `Student_Depression_Dataset.csv` để thực hiện phân tích dữ liệu và thử nghiệm các mô hình học máy phân loại nhị phân nhãn `Depression` gồm `Yes` và `No`.

Mục tiêu:
- Khám phá cấu trúc và chất lượng dữ liệu.
- Phân tích phân bố nhãn `Depression` và các thuộc tính.
- Trực quan hóa dữ liệu, quan sát các mối liên hệ thống kê.
- Tiền xử lý dữ liệu và xây dựng mô hình Machine Learning.
- Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score, Confusion Matrix và ROC-AUC.
- Phân tích mức độ quan trọng của thuộc tính.
- Thảo luận giới hạn, nguy cơ thiên lệch, rò rỉ dữ liệu và khả năng khái quát.

**Bài toán:** Supervised Learning – Binary Classification  
**Biến mục tiêu:** `Depression`  
**Nhãn:** `Yes`, `No`

> **Lưu ý:** Đây là dự án học tập về phân tích dữ liệu và Machine Learning. Nhãn trong bộ dữ liệu không được xem là kết luận chẩn đoán y khoa. Mô hình không được sử dụng để tự chẩn đoán hoặc đưa ra quyết định chăm sóc sức khỏe cá nhân.

## 2. Thông tin bộ dữ liệu

### 2.1. Tổng quan

File dữ liệu được kiểm tra ban đầu có **4.464 dòng và 13 cột**. Tại thời điểm kiểm tra ban đầu, dữ liệu không có giá trị thiếu và không có dòng trùng lặp hoàn toàn. Notebook sẽ kiểm tra lại các thông tin này mỗi lần chạy.

| Thông tin | Giá trị |
|---|---:|
| Số dòng ban đầu | 4.464 |
| Số thuộc tính | 13 |
| Số thuộc tính đầu vào | 12 |
| Biến mục tiêu | 1 (`Depression`) |
| Nhãn mục tiêu | `Yes`, `No` |
| Giá trị thiếu ở lần kiểm tra ban đầu | 0 |
| Dòng trùng lặp ở lần kiểm tra ban đầu | 0 |

### 2.2. Mô tả các thuộc tính

Các mô tả dưới đây chỉ là diễn giải sơ bộ dựa trên tên cột. Những trường chưa có tài liệu định nghĩa gốc cần được xác minh trước khi đưa ra kết luận chuyên môn.

| Cột | Mô tả sơ bộ | Ghi chú |
|---|---|---|
| `Gender` | Nhóm giới tính được ghi trong dữ liệu | Không suy diễn ngoài các giá trị có trong file |
| `Age` | Tuổi | Thuộc tính số |
| `Family_background` | Nhóm hoàn cảnh gia đình | Căn cứ vào các nhóm giá trị trong dữ liệu |
| `Relationship_status` | Trạng thái mối quan hệ | Không diễn giải thành quan hệ nhân quả |
| `Marital_status` | Tình trạng hôn nhân | Thuộc tính phân loại |
| `Living_style` | Hình thức sinh sống | Cần xác minh định nghĩa gốc nếu phân tích chuyên sâu |
| `Current_semester` | Học kỳ hiện tại | Cần kiểm tra ý nghĩa các giá trị đặc biệt |
| `CGPA` | Điểm trung bình tích lũy | Cần xác minh thang điểm |
| `friend_circle` | Thuộc tính Yes/No liên quan đến bạn bè | Cần xác minh câu hỏi khảo sát gốc |
| `Health_condition` | Nhóm tình trạng sức khỏe theo nhãn trong dữ liệu | Cần xác minh định nghĩa và cách thu thập |
| `Cost_without_study` | Trường dữ liệu về chi phí, chưa rõ định nghĩa cụ thể | Không tự diễn giải khi chưa có tài liệu gốc |
| `Home_division` | Khu vực/đơn vị địa lý trong bộ dữ liệu | Giữ nguyên cách phân loại của dữ liệu |
| `Depression` | Nhãn mục tiêu `Yes` hoặc `No` | Không mặc định là chẩn đoán lâm sàng |

## 3. Cấu trúc thư mục dự án

Cấu trúc đề xuất:

```text
BaoCaoCuoiKy/
├── README.md
├── notebooks/
│   └── PhatHienTramCamSinhVien.ipynb
├── requirements.txt
├── .gitignore
└── data/
    └── Student_Depression_Dataset.csv
```

- `README.md`: tài liệu giới thiệu, hướng dẫn cài đặt và chạy dự án.
- `notebooks/PhatHienTramCamSinhVien.ipynb`: mã nguồn, phân tích, biểu đồ, huấn luyện và đánh giá mô hình.
- `requirements.txt`: danh sách thư viện Python.
- `.gitignore`: các tệp/thư mục không cần đưa lên Git.
- `data/Student_Depression_Dataset.csv`: bộ dữ liệu đầu vào.

Nếu đặt CSV ở vị trí khác, cần cập nhật đường dẫn đọc dữ liệu trong Notebook. Không đưa dữ liệu lên kho Git công khai nếu chưa kiểm tra quyền chia sẻ và giấy phép.

## 4. Yêu cầu môi trường và cài đặt

- Python 3.10 trở lên (khuyến nghị).
- Jupyter Notebook hoặc Visual Studio Code có hỗ trợ Jupyter.
- Bộ dữ liệu CSV đặt đúng vị trí Notebook có thể đọc.

### 4.1. Tạo môi trường ảo trên Windows

Mở Terminal hoặc PowerShell tại thư mục gốc dự án:

```powershell
python -m venv .venv
```

Kích hoạt bằng PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn chạy script, có thể dùng Git Bash:

```bash
source .venv/Scripts/activate
```

Hoặc Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 4.2. Cài đặt thư viện

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu chưa có `requirements.txt`, tạo file với nội dung ban đầu:

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

## 5. Hướng dẫn chạy Notebook

1. Tải hoặc clone dự án về máy.
2. Đặt CSV vào `data/Student_Depression_Dataset.csv`.
3. Mở thư mục dự án bằng VS Code hoặc khởi động Jupyter.
4. Kích hoạt môi trường `.venv`.
5. Cài thư viện trong `requirements.txt`.
6. Mở `notebooks/PhatHienTramCamSinhVien.ipynb`.
7. Chọn kernel Python thuộc môi trường `.venv`.
8. Chạy lần lượt các cell hoặc chọn **Restart Kernel and Run All**.
9. Kiểm tra kết quả, biểu đồ và nhận xét trong Notebook.

**Lưu ý:** Với cấu trúc đề xuất, đường dẫn CSV thường là `data/Student_Depression_Dataset.csv`. Nếu tên file hoặc vị trí khác, hãy sửa biến đường dẫn trong phần đọc dữ liệu của Notebook.

## 6. Quy trình thực hiện

| Giai đoạn | Nội dung |
|---|---|
| 1. Khởi tạo | Giới thiệu đề tài, mục tiêu và import thư viện |
| 2. Đọc dữ liệu | Đọc CSV, xem kích thước và một số dòng dữ liệu |
| 3. Khảo sát dữ liệu | Kiểm tra kiểu dữ liệu, thống kê mô tả và giá trị duy nhất |
| 4. Kiểm tra chất lượng | Kiểm tra giá trị thiếu, dòng trùng lặp và giá trị bất thường |
| 5. EDA | Phân tích phân bố nhãn, biểu đồ biến số/biến phân loại và so sánh theo nhãn |
| 6. Tiền xử lý | Tách X/y, chia train/test, điền thiếu, chuẩn hóa và mã hóa |
| 7. Huấn luyện | Dummy Classifier, Logistic Regression và Random Forest |
| 8. Đánh giá | Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix và Cross-validation |
| 9. Phân tích mô hình | Permutation Importance và thử nghiệm dự đoán |
| 10. Kết luận | Thảo luận kết quả, hạn chế và hướng phát triển |

## 7. Tiền xử lý và xây dựng mô hình

### 7.1. Phân loại thuộc tính

- **Thuộc tính số:** `Age`, `Current_semester`, `CGPA`, `Cost_without_study`.
- **Thuộc tính phân loại:** `Gender`, `Family_background`, `Relationship_status`, `Marital_status`, `Living_style`, `friend_circle`, `Health_condition`, `Home_division`.

### 7.2. Tiền xử lý

- Thuộc tính số: điền giá trị thiếu bằng trung vị khi cần và chuẩn hóa bằng `StandardScaler`.
- Thuộc tính phân loại: điền giá trị thiếu bằng giá trị phổ biến nhất khi cần và mã hóa bằng `OneHotEncoder`.
- Chia dữ liệu thành tập huấn luyện và tập kiểm tra theo tỷ lệ 80:20.
- Sử dụng chia dữ liệu phân tầng theo biến mục tiêu để giữ tỷ lệ nhãn tương đối ổn định giữa hai tập.

Các bước tiền xử lý được tổ chức trong pipeline để hạn chế việc học thông tin từ tập kiểm tra trong quá trình biến đổi dữ liệu.

### 7.3. Các mô hình

- **DummyClassifier:** mô hình cơ sở (baseline), dùng làm mốc so sánh.
- **Logistic Regression:** mô hình phân loại tuyến tính.
- **Random Forest:** mô hình tổ hợp nhiều cây quyết định.

Notebook sử dụng trọng số lớp `balanced` cho Logistic Regression và Random Forest nhằm lưu ý đến sự chênh lệch số lượng giữa hai nhãn.

## 8. Chỉ số đánh giá

| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy | Tỷ lệ dự đoán đúng trên tổng số mẫu |
| Precision (Yes) | Trong các mẫu được dự đoán `Yes`, tỷ lệ mẫu thực sự mang nhãn `Yes` |
| Recall (Yes) | Trong các mẫu thực tế mang nhãn `Yes`, tỷ lệ được mô hình nhận diện |
| F1-score (Yes) | Trung bình điều hòa giữa Precision và Recall của lớp `Yes` |
| ROC-AUC | Khả năng phân biệt hai lớp theo các ngưỡng dự đoán |
| Confusion Matrix | Số lượng dự đoán đúng và sai của từng lớp |
| Stratified 5-Fold Cross-validation | Đánh giá độ ổn định qua nhiều lần chia tập huấn luyện có phân tầng |

Do dữ liệu có tỷ lệ hai nhãn không cân bằng hoàn toàn, không nên chỉ dựa vào Accuracy. Cần xem thêm Precision, Recall và F1-score của lớp `Yes`.

## 9. Kết quả thực nghiệm

### 9.1. Thống kê dữ liệu

Theo lần kiểm tra ban đầu:

| Nội dung | Kết quả |
|---|---:|
| Số dòng | 4.464 |
| Số thuộc tính | 13 |
| Nhãn `No` | 3.263 (khoảng 73,1%) |
| Nhãn `Yes` | 1.201 (khoảng 26,9%) |
| Giá trị thiếu | 0 |
| Dòng trùng lặp hoàn toàn | 0 |

Tập dữ liệu được chia thành:
- Tập huấn luyện: 3.571 mẫu.
- Tập kiểm tra: 893 mẫu.

### 9.2. So sánh mô hình trên tập kiểm tra

Các số liệu dưới đây là kết quả của lần chạy Notebook đã được cung cấp. Khi chạy lại Notebook với dữ liệu hoặc cấu hình khác, cần cập nhật bảng theo kết quả mới.

| Mô hình | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 73,1% | 0% | 0% | 0% | 0,500 |
| Logistic Regression | 96,9% | 92,7% | 95,8% | 94,3% | 0,993 |
| Random Forest | **99,1%** | **99,6%** | **97,1%** | **98,3%** | **0,999** |

Theo kết quả lần chạy này, Random Forest có F1-score lớp `Yes` cao nhất trong ba mô hình được so sánh.

### 9.3. Stratified 5-Fold Cross-validation

| Mô hình | Accuracy TB | Precision TB | Recall TB | F1 TB | ROC-AUC TB |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 73,1% | 0% | 0% | 0% | 0,500 |
| Logistic Regression | 96,5% | 91,3% | 96,3% | 93,7% | 0,991 |
| Random Forest | **98,9%** | **99,2%** | **96,8%** | **97,9%** | **0,999** |

Kết quả Cross-validation cho thấy Random Forest tiếp tục có các chỉ số trung bình cao trong lần chạy được ghi nhận. Đây vẫn là kết quả trên cùng bộ dữ liệu, không thay thế việc kiểm tra trên một bộ dữ liệu độc lập.

### 9.4. Ma trận nhầm lẫn của Random Forest

Trên tập kiểm tra gồm 893 mẫu:

| Nhãn thực tế | Dự đoán No | Dự đoán Yes |
|---|---:|---:|
| No | 652 | 1 |
| Yes | 7 | 233 |

Diễn giải:
- 652 mẫu `No` được dự đoán đúng.
- 233 mẫu `Yes` được dự đoán đúng.
- 1 mẫu `No` bị dự đoán nhầm thành `Yes`.
- 7 mẫu `Yes` bị dự đoán nhầm thành `No`.

### 9.5. Permutation Importance

Kết quả trong lần chạy được ghi nhận:

| Thứ tự | Thuộc tính | Importance mean |
|---:|---|---:|
| 1 | `Health_condition` | 0,663849 |
| 2 | `Relationship_status` | 0,122807 |
| 3 | `Marital_status` | 0,049775 |
| 4 | `friend_circle` | 0,039858 |
| 5 | `CGPA` | 0,029297 |
| 6 | `Family_background` | 0,027013 |
| 7 | `Current_semester` | 0,020547 |
| 8 | `Cost_without_study` | 0,006356 |
| 9 | `Living_style` | 0,002309 |
| 10 | `Gender` | 0,000860 |
| 11 | `Home_division` | 0,000400 |
| 12 | `Age` | 0,000000 |

`Health_condition` có mức độ quan trọng nổi bật so với các thuộc tính còn lại. Kết quả này cần được diễn giải thận trọng và phải kiểm tra nguy cơ rò rỉ dữ liệu trước khi kết luận.

## 10. Hạn chế và vấn đề cần kiểm tra

### 10.1. Nguy cơ rò rỉ dữ liệu từ `Health_condition`

Trong lần phân tích được ghi nhận, tỷ lệ nhãn `Depression` có sự phân tách rất rõ theo `Health_condition`:
- Nhóm `Critical`: 100% nhãn `Yes`.
- Nhóm `Good`: 100% nhãn `No`.

Đồng thời, `Health_condition` có Permutation Importance cao nhất. Đây là dấu hiệu cần kiểm tra vì thuộc tính này có thể chứa thông tin liên quan trực tiếp đến cách tạo nhãn mục tiêu hoặc được thu thập sau khi đã xác định nhãn.

Chưa thể kết luận chắc chắn có data leakage nếu chưa biết quy trình thu thập và gán nhãn của bộ dữ liệu. Cần xác minh định nghĩa, thời điểm thu thập và nguồn gốc của thuộc tính này.

**Thí nghiệm nên thực hiện tiếp theo:**
1. Giữ kết quả mô hình hiện tại làm kết quả ban đầu.
2. Loại bỏ `Health_condition` khỏi tập thuộc tính đầu vào.
3. Huấn luyện và đánh giá lại các mô hình.
4. So sánh Accuracy, Precision, Recall, F1-score và ROC-AUC trước/sau khi loại bỏ thuộc tính.
5. Chọn mô hình dựa trên Cross-validation của tập huấn luyện; chỉ sử dụng tập kiểm tra để đánh giá cuối cùng.

Không nên khẳng định mô hình có khả năng khái quát cao chỉ dựa trên kết quả hiện tại.

### 10.2. Giới hạn của dữ liệu

- Chưa có đủ thông tin để xác minh toàn bộ cách thu thập và tiêu chí gán nhãn.
- Một số tên thuộc tính chưa có định nghĩa chi tiết.
- Phân bố nhãn `Yes` và `No` chưa cân bằng.
- Dữ liệu có thể không đại diện cho toàn bộ sinh viên hoặc các nhóm dân số khác.
- Mối liên hệ thống kê không chứng minh quan hệ nhân quả.
- Kết quả trên một bộ dữ liệu không bảo đảm hiệu quả trên dữ liệu thực tế khác.

### 10.3. Giới hạn sử dụng

Mô hình chỉ phục vụ mục đích học tập và minh họa kỹ thuật phân tích dữ liệu. Không sử dụng kết quả dự đoán để chẩn đoán trầm cảm, tự đánh giá tình trạng sức khỏe hoặc thay thế tư vấn của chuyên gia.

## 11. Hướng phát triển

- Xác minh nguồn dữ liệu, định nghĩa thuộc tính và quy trình gán nhãn.
- Kiểm tra và xử lý các thuộc tính có nguy cơ data leakage.
- Thử nghiệm mô hình khi loại bỏ `Health_condition`.
- Bổ sung dữ liệu có nguồn gốc rõ ràng và tính đại diện tốt hơn.
- Đánh giá mô hình trên tập dữ liệu độc lập.
- Phân tích hiệu suất theo từng nhóm dữ liệu khi có đủ cơ sở và bảo đảm quyền riêng tư.
- Hoàn thiện tài liệu mô tả dữ liệu, cấu hình thí nghiệm và khả năng tái lập kết quả.

## 12. Nguồn dữ liệu và tài liệu tham khảo

Thông tin nguồn dữ liệu được cung cấp cho dự án:
- **Tên file:** `Student_Depression_Dataset.csv`
- **URL nguồn:** https://doi.org/10.34740/kaggle/dsv/20867505
- **Tác giả/đơn vị:** Obaidul Haque
- **Ngày truy cập được ghi nhận:** 01/11/2026
- **Giấy phép được ghi nhận:** CC BY-SA 4.0

Cần đối chiếu lại trang phát hành dữ liệu để xác nhận tác giả, phiên bản, giấy phép và điều kiện sử dụng trước khi công khai hoặc phân phối lại dữ liệu. Thông tin trên được ghi theo dữ liệu nguồn do người thực hiện cung cấp, chưa thay thế việc xác minh trực tiếp giấy phép.

**Tác giả:** `Lê Tuấn Anh`  
**Lớp:** `DHKTPM20B`  
**Môn học:** `Nhập môn phân tích dữ liệu & AI`  
**Trường:** `Đại học Công nghiệp TPHCM`  
**Năm học:** `2026-2027`

> Dự án được thực hiện với mục đích học tập và nghiên cứu về phân tích dữ liệu và Machine Learning. Các kết quả cần được diễn giải trong phạm vi bộ dữ liệu và những giới hạn đã nêu.
