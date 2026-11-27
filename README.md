# Nhận Diện Cảm Xúc Khuôn Mặt với CNN (FER2013)

Dự án xây dựng và huấn luyện mô hình Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN) bằng thư viện PyTorch để phân loại 7 trạng thái cảm xúc cơ bản của con người trên bộ dữ liệu FER2013.

## Thành viên nhóm & Phân công công việc
|STT|Thành viên|Vai trò & Nhiệm vụ chính|File phụ trách|
|-----|-----|-------|------|
|1|Như Ý|Tiền xử lý dữ liệu, Data Augmentation, xây dựng DataLoader và trực quan hóa phân bố dữ liệu|src/01_dataset.ipynb|
|2|Văn Hậu|Thiết kế kiến trúc mạng EmotionCNN, vẽ sơ đồ kiến trúc và quản lý lưu/nạp mô hình|src/02_model.ipynb|
|3|Phương Vy|Xây dựng vòng lặp huấn luyện (Train/Val), tối ưu hóa (Adam, ReduceLROnPlateau), lưu checkpoint & vẽ Learning Curves|src/03_train.ipynb|
|4|Minh Thùy|Đánh giá mô hình trên tập Test, vẽ Confusion Matrix, xuất Classification Report và trực quan hóa Conv Filters|src/04_evaluate.ipynb|

## Cấu trúc thư mục dự án
```
fer_project/
│
├── data/                               # [Như Ý phụ trách] Chứa dữ liệu
│   └── fer2013.csv                     # File dữ liệu tải từ Kaggle đặt ở đây
│
├── models/                             # [Văn Hậu & Phương Vy] Lưu trọng số mô hình đã train
│   ├── best_model.pth                  # Checkpoint có validation accuracy cao nhất
│   └── last_model.pth                  # Checkpoint ở epoch cuối cùng
│
├── outputs/                            # [Phương Vy] Lưu kết quả log thí nghiệm
│   └── training_history.csv            # File CSV lưu loss, accuracy qua từng epoch
│
├── figures/                            # [Minh Thùy & Như Ý] Lưu toàn bộ hình ảnh, biểu đồ
│   ├── dataset_distribution.png        # Biểu đồ phân bố 7 lớp cảm xúc (Như Ý)
│   ├── loss_acc_curves.png             # Biểu đồ học tập loss/acc (Phương Vy)
│   ├── confusion_matrix.png            # Ma trận nhầm lẫn (Minh Thùy)
│   └── conv_filters.png                # Hình trực quan hóa các bộ lọc tích chập (Minh Thùy)
│
├── src/                                # Thư mục chứa mã nguồn chính 
│   ├── dataset.ipynb                   # [Như Ý] Xử lý dữ liệu & Data Augmentation
│   ├── model.ipynb                     # [Văn Hậu] Đnh nghĩa kiến trúc mạng CNN
│   ├── train.ipynb                     # [Phương Vy] Vòng lặp huấn luyện, tối ưu & lưu log
│   └── evaluate.ipynb                  # [Minh Thùy] Đánh giá, Confusion Matrix & Visualize Filters
│
├── FER_CNN_.ipynb                      # Notebook chạy chung lúc đầu để cả nhóm test
├── requirements.txt                    # Danh sách các thư viện cần cài đặt (torch, pandas, seaborn,...)
└── README.md                           # Tài liệu hướng dẫn sử dụng và giới thiệu đồ án của nhóm
```

## Giới thiệu bộ dữ liệu FER2013

- Nguồn dữ liệu: FER2013 trên Kaggle (dạng file fer2013.csv).
-  Đặc điểm ảnh: Ảnh xám (Grayscale) kích thước $48 \times 48$ pixels.
-  Số lượng lớp (7 cảm xúc):
  ```0: Angry | 1: Disgust | 2: Fear | 3: Happy | 4: Sad | 5: Surprise | 6: Neutral```
- Phân chia tập dữ liệu:Training: 28,709 ảnhPublicTest (Validation): 3,589 ảnhPrivateTest (Test): 3,589 ảnh

## Kiến trúc mô hình (EmotionCNN)
Mô hình được thiết kế gọn nhẹ, tối ưu cho ảnh kích thước $48 \times 48 \times 1$:
- Feature Extractor: Gồm 2 khối tích chập liên tiếp (Conv2d $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ MaxPool2d $\rightarrow$ Dropout) giúp trích xuất đặc trưng không gian từ cơ bản (cạnh, đường nét) đến phức tạp (vùng mắt, miệng).
- Classifier: Các lớp Linear kết hợp ReLU và Dropout để phân loại ra 7 xác suất cảm xúc tương ứng.

## Hướng dẫn cài đặt và chạy dự án
Bước 1: Cài đặt thư viện
Đảm bảo máy đã cài đặt Python $\ge 3.8$, sau đó chạy lệnh sau tại thư mục gốc fer_project/:
```
pip install -r requirements.txt
```
Bước 2: Chuẩn bị dữ liệu
Tải file fer2013.csv và đặt vào đúng đường dẫn:
```
fer_project/data/fer2013.csv
```
Bước 3: Thực thi chương trình
Bạn có thể chọn một trong hai cách sau:
- Cách 1 — Chạy toàn bộ quy trình (Khuyên dùng):
Mở file FER_CNN.ipynb bằng Jupyter Notebook / VS Code / Google Colab và chọn Run All để chạy lần lượt từ tiền xử lý, huấn luyện (15 epochs) đến đánh giá và xuất biểu đồ.
- Cách 2 — Chạy từng phần theo từng thành viên:
  Chạy tuần tự các notebook trong thư mục src/:
  1. src/dataset.ipynb
  2. src/model.ipynb
  3. src/train.ipynb (tạo ra models/best_model.pth)
  4. src/evaluate.ipynb (sử dụng models/best_model.pth để đánh giá)
Mẹo tối ưu tốc độ: Nếu chạy trên Google Colab, hãy bật GPU (Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU) để rút ngắn thời gian huấn luyện 15 epochs xuống chỉ còn vài phút.

## Kết quả trực quan hóa đầu ra
Sau khi hoàn tất chạy notebook, toàn bộ kết quả sẽ tự động được lưu gọn gàng vào các thư mục:
- Mô hình tốt nhất: models/best_model.pth
- Biểu đồ đánh giá: figures/loss_acc_curves.png, figures/confusion_matrix.png, figures/conv1_filters.png
- Số liệu chi tiết: outputs/training_history.csv, outputs/classification_report.txt
