# FER2013 Facial Expression Recognition (`fer_project`) — Project Template

Tài liệu mẫu cấu trúc dự án, phân công công việc và quy ước làm việc cho nhóm thực hiện bài toán **Nhận diện cảm xúc khuôn mặt trên tập dữ liệu FER2013** bằng mạng nơ-ron tích chập (CNN) với PyTorch.

```
fer_project/
│
├── data/                               # [Minh Thùy phụ trách] Chứa dữ liệu
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
│   ├── dataset_distribution.png        # Biểu đồ phân bố 7 lớp cảm xúc (Minh Thùy)
│   ├── loss_acc_curves.png             # Biểu đồ học tập loss/acc (Phương Vy)
│   ├── confusion_matrix.png            # Ma trận nhầm lẫn (Như Ý)
│   └── conv_filters.png                # Hình trực quan hóa các bộ lọc tích chập (Như Ý)
│
├── src/                                # Thư mục chứa mã nguồn chính 
│   ├── dataset.ipynb                   # [Minh Thùy] Xử lý dữ liệu & Data Augmentation
│   ├── model.ipynb                     # [Văn Hậu] Đnh nghĩa kiến trúc mạng CNN
│   ├── train.ipynb                     # [Phương Vy] Vòng lặp huấn luyện, tối ưu & lưu log
│   └── evaluate.ipynb                  # [Như Ý] Đánh giá, Confusion Matrix & Visualize Filters
│
├── FER_CNN_.ipynb                      # Notebook chạy chung lúc đầu để cả nhóm test
├── requirements.txt                    # Danh sách các thư viện cần cài đặt (torch, pandas, seaborn,...)
└── README.md                           # Tài liệu hướng dẫn sử dụng và giới thiệu đồ án của nhóm
```
## Phân công nhiệm vụ & Sản phẩm đầu ra
|Thành viên|	Phụ trách chính	|Notebook trong src/	|Sản phẩm đầu ra (models/, outputs/, figures/)|
|------|--------|-------|------|
|Minh Thùy|	Đọc dữ liệu, tiền xử lý, Data Augmentation, kiểm tra phân bố nhãn	|src/dataset.ipynb|	data/fer2013.csv, figures/dataset_distribution.png|
|Văn Hậu|	Thiết kế kiến trúc CNN (EmotionCNN), sơ đồ kiến trúc|	src/model.ipynb|	figures/model_architecture.png|
|Phương Vy	|Vòng lặp Train/Val, Optimizer, LR Scheduler, ghi log & vẽ learning curves|	src/train.ipynb	|models/best_model.pth, models/last_model.pth, outputs/training_history.csv, figures/loss_acc_curves.png|
|Như Ý	|Đánh giá tập Test, Confusion Matrix, Classification Report, trực quan hóa Conv Filters	|src/evaluate.ipynb	|outputs/classification_report.txt, figures/confusion_matrix.png, figures/conv1_filters.png|

## Quy ước kỹ thuật & Lưu ý khi làm việc với Notebook
- Tránh xung đột tên Image: Khi dùng cả PIL và IPython.display trong notebook, luôn import rõ ràng:

```
from PIL import Image as PILImage
from IPython.display import Image, display
```

- Thứ tự chạy & phụ thuộc giữa các Notebook:

    1. **dataset.ipynb:** Kiểm tra dữ liệu và xuất biểu đồ phân bố lớp.

    2. **model.ipynb:** Kiểm tra cấu trúc EmotionCNN, shape đầu vào/đầu ra và xuất sơ đồ kiến trúc.

    3. **train.ipynb:** Huấn luyện mô hình và xuất trọng số best_model.pth + lịch sử train.

    4. **evaluate.ipynb:** Nạp best_model.pth để đánh giá tập Test, vẽ Confusion Matrix và trích xuất Conv Filters.

    5. **FER_CNN.ipynb:** Ghép toàn bộ 4 phần theo đúng thứ tự trên để nộp bản hoàn chỉnh.