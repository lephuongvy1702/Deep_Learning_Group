# Xấp xỉ hàm mật độ Gaussian 2D bằng mạng MLP (PyTorch)

Dự án sử dụng mạng nơ-ron truyền thẳng **Multilayer Perceptron (MLP)** xây dựng bằng PyTorch để xấp xỉ hàm mật độ xác suất Gaussian hai chiều.

Notebook chính: `ex (14)(1).ipynb` (nên đổi tên thành `main.ipynb` khi đóng gói dự án và cập nhật tên tương ứng trong tài liệu).

## 1. Nội dung dự án

Notebook triển khai các bước:

1. Sinh dữ liệu từ hàm mật độ Gaussian 2D.
2. Lưu và đọc dữ liệu CSV.
3. Thống kê mô tả và trực quan hóa dữ liệu.
4. Chuẩn hóa hai đặc trưng đầu vào bằng Min-Max.
5. Xây dựng và huấn luyện các mô hình MLP.
6. Khảo sát ảnh hưởng của quy mô dữ liệu, số nơ-ron và số tầng ẩn.
7. Đánh giá mô hình bằng MSE, RMSE và MAE.
8. Lưu checkpoint, lịch sử huấn luyện, biểu đồ và kết quả dự đoán.

## 2. Sinh dữ liệu

Hai đầu vào được lấy mẫu độc lập theo phân phối đều:

- \(x_1 \sim U(-2, 2)\)
- \(x_2 \sim U(-4, 4)\)

Giá trị mục tiêu \(y\) được tính bằng mật độ Gaussian hai chiều với:

```python
mean = [0.0, 0.0]
cov = [[0.3, -0.5],
       [-0.5, 2.0]]
```

Công thức:

\[
f(x)=\frac{1}{2\pi\sqrt{|C|}}
\exp\left(-\frac{1}{2}(x-\mu)^T C^{-1}(x-\mu)\right)
\]

Trong đó \(\mu\) là vector trung bình, \(C\) là ma trận hiệp phương sai, \(|C|\) là định thức của \(C\), và \(C^{-1}\) là ma trận nghịch đảo.

### Các tập dữ liệu

| Tập | Số mẫu | Seed sinh dữ liệu |
|---|---:|---:|
| Training | 20.000 | 42 |
| Testing | 100 | 43 |
| Validation | 100 | 44 |

Dữ liệu được lưu tại:

- `data/train.csv`
- `data/test.csv`
- `data/valid.csv`

Mỗi tệp gồm ba cột: `x1`, `x2`, `y`. Notebook tự sinh và ghi đè các tệp CSV này khi chạy cell tạo dữ liệu.

**Lưu ý:** tập Validation và Test chỉ có 100 mẫu mỗi tập, nhỏ hơn nhiều so với tập Training. Vì vậy, các chỉ số đánh giá có thể nhạy với cách lấy mẫu.

## 3. Tiền xử lý

Notebook định nghĩa `MinMaxScalerCustom` để chuẩn hóa hai đầu vào về khoảng gần \([0,1]\):

\[
x_{\text{scaled}} =
\frac{x-x_{\min}}{x_{\max}-x_{\min}+10^{-8}}
\]

Scaler được fit trên Training; các giá trị min/max đó được dùng để biến đổi Validation, Test và các tọa độ dự đoán mới. Biến mục tiêu `y` không được chuẩn hóa.

## 4. Mô hình

### 4.1. `RegressionMLP`

Mô hình nhận hai đầu vào và trả về một giá trị dự đoán:

- Tầng đầu vào: 2 đặc trưng.
- Mỗi tầng ẩn: `Linear` + `Sigmoid`.
- Tầng đầu ra: `Linear`, không có Sigmoid.

Kiến trúc tổng quát:

```text
2 → H → ... → H → 1
```

Với cấu hình cơ sở `H = 10`, `C = 1`, mô hình có 41 tham số có thể huấn luyện.

### 4.2. `DeepRegressionMLP` trong Experiment 3

Trong mã nguồn notebook hiện tại, `DeepRegressionMLP` nối các tầng `Linear` liên tiếp nhưng **không có hàm kích hoạt giữa các tầng**. Do đó, chuỗi tầng này vẫn chỉ biểu diễn một phép biến đổi affine; tăng số tầng Linear không tạo thêm tính phi tuyến.

Vì vậy, Experiment 3 hiện khảo sát số tầng Linear, chưa phải một phép so sánh đầy đủ về độ sâu của MLP phi tuyến. Nếu muốn nghiên cứu ảnh hưởng của độ sâu MLP, cần thêm activation (ví dụ `Sigmoid` hoặc `ReLU`) giữa các tầng và chạy lại thí nghiệm. Kết quả cũ không được dùng làm kết quả cho kiến trúc đã sửa.

## 5. Thiết kế thí nghiệm

Các cấu hình được chạy với ba seed huấn luyện mô hình: `42`, `123`, `2026`.

| Thí nghiệm | Yếu tố khảo sát | Giá trị khảo sát | Yếu tố cố định |
|---|---|---|---|
| Experiment 1 | Số mẫu Training \(A\) | 200, 400, ..., 2000 (bước 200) | `H = 10`, một tầng ẩn |
| Experiment 2 | Số nơ-ron tầng ẩn \(H\) | 1 đến 10 | 200 mẫu Training, một tầng ẩn |
| Experiment 3 | Số tầng Linear ẩn \(C\) | 1 đến 4 | 200 mẫu Training, 10 nơ-ron mỗi tầng |

Validation và Test được tái sử dụng giữa các cấu hình. Trong mã nguồn hiện tại, Test được tính chỉ số cho từng cấu hình trong các bảng thí nghiệm; vì vậy Test không hoàn toàn được giữ riêng cho một lần đánh giá cuối cùng. Khi diễn giải kết quả, cần lưu ý điều này.

## 6. Thiết lập huấn luyện

Các thí nghiệm sử dụng:

| Thành phần | Giá trị |
|---|---|
| Framework | PyTorch |
| Loss | `nn.MSELoss()` |
| Optimizer | SGD |
| Learning rate | `0.01` |
| Epoch tối đa | `200` |
| Batch size | `32` |
| Shuffle Training | Có |
| Shuffle Validation | Không |
| Thiết bị | CUDA nếu khả dụng, nếu không dùng CPU |

Sau mỗi epoch, mô hình được đánh giá trên Validation. Trọng số tại epoch có Validation Loss thấp nhất được giữ lại cho lần chạy đó. Notebook đặt seed cho Python `random`, NumPy và PyTorch; đồng thời cấu hình cuDNN deterministic khi CUDA khả dụng.

## 7. Đánh giá

Các chỉ số được sử dụng:

- **MSE (Mean Squared Error):** trung bình bình phương sai số.
- **RMSE (Root Mean Squared Error):** căn bậc hai của MSE.
- **MAE (Mean Absolute Error):** trung bình trị tuyệt đối sai số.

Notebook tạo đường cong Training/Validation Loss, đồ thị sai số theo từng yếu tố khảo sát, biểu đồ giá trị thực tế so với dự đoán và bề mặt Gaussian dự đoán so với bề mặt thật.

## 8. Kết quả có trong output notebook

Các số liệu dưới đây được lấy từ output đã lưu trong notebook đính kèm. Đây là kết quả của lần chạy đó, không đảm bảo giống hệt khi chạy lại trên phần cứng hoặc môi trường khác.

### 8.1. Cấu hình checkpoint được lưu

| Thí nghiệm | Cấu hình được lưu | Validation Loss |
|---|---|---:|
| Experiment 1 | 2000 mẫu Training, `H = 10` | 0.003232 |
| Experiment 2 | `H = 4`, 200 mẫu Training | 0.003259 |
| Experiment 3 | `C = 4`, 10 nơ-ron/tầng, 200 mẫu Training | 0.003250 |

Trong code, checkpoint của mỗi thí nghiệm được chọn bằng cách lấy lần chạy có `val_loss` nhỏ nhất trong danh sách kết quả của thí nghiệm đó. Đây là lựa chọn theo **một lần chạy/seed tốt nhất**, không phải trung bình Validation Loss của ba seed.

### 8.2. So sánh ba mô hình trong phần Prediction

| Mô hình | Test MSE | Test RMSE | Test MAE |
|---|---:|---:|---:|
| Experiment 1 | 0.002775 | 0.052682 | 0.039115 |
| Experiment 2 | 0.002805 | 0.052959 | 0.039330 |
| Experiment 3 | 0.002796 | 0.052882 | 0.039425 |

Theo bảng so sánh trong output, Experiment 1 có các chỉ số sai số thấp nhất trong ba checkpoint được so sánh. Phần đánh giá mô hình tốt nhất ghi nhận:

- Thí nghiệm: Experiment 1
- Cấu hình: 2000 mẫu Training
- Validation Loss: `0.00323244`
- Test MSE: `0.00277542`
- Test RMSE: `0.05268228`
- Test MAE: `0.03911523`

Đây là kết quả của checkpoint và lần chạy được lưu trong notebook, không phải kết luận rằng Experiment 1 luôn tốt nhất trong mọi lần chạy.

### 8.3. Dự đoán tại tọa độ mới

Notebook minh họa dự đoán tại ba tọa độ:

- `(0.0, 0.0)`
- `(0.5, 1.0)`
- `(-1.0, 2.0)`

Kết quả được ghi vào `mlp_outputs/new_point_predictions.csv`.

## 9. Cấu trúc thư mục

Sau khi chạy notebook, cấu trúc đầu ra dự kiến như sau:

```text
project/
├── ex (14)(1).ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── valid.csv
├── checkpoints/
│   ├── exp1_best.pth
│   ├── exp2_best.pth
│   └── exp3_best.pth
└── mlp_outputs/
    ├── run_<timestamp>/
    │   ├── checkpoints/
    │   ├── figures/
    │   └── logs/
    ├── figures/
    │   ├── actual_vs_predicted.png
    │   └── predicted_gaussian_surfaces.png
    ├── test_predictions.csv
    ├── prediction_metrics.csv
    └── new_point_predictions.csv
```

Trong đó:

- `checkpoints/`: checkpoint tổng hợp của từng thí nghiệm.
- `mlp_outputs/run_<timestamp>/checkpoints/`: checkpoint của từng cấu hình và seed trong một lần chạy.
- `mlp_outputs/run_<timestamp>/logs/`: log và lịch sử huấn luyện JSON.
- `mlp_outputs/run_<timestamp>/figures/`: biểu đồ của các thí nghiệm.
- `mlp_outputs/`: các bảng dự đoán và biểu đồ so sánh ở phần cuối notebook.

Notebook hiện cũng lưu một số biểu đồ ở thư mục làm việc hiện tại thay vì thư mục `mlp_outputs`; hãy kiểm tra cell lưu hình nếu muốn gom toàn bộ biểu đồ về một nơi.

## 10. Cài đặt môi trường

Nên sử dụng môi trường Python riêng. Notebook đính kèm có metadata Python `3.13.10`; phiên bản PyTorch phù hợp cần được cài theo hệ điều hành và CPU/CUDA của máy.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu cần GPU, cài PyTorch theo hướng dẫn cài đặt chính thức phù hợp với phiên bản CUDA của máy. Nếu không, có thể dùng bản CPU.

**Graphviz:** `torchviz` được import trong notebook. Nếu sử dụng chức năng render đồ thị tính toán của `torchviz`, cần cài thêm Graphviz ở cấp hệ điều hành và bảo đảm chương trình `dot` có trong `PATH`.

## 11. Cách chạy

1. Mở terminal tại thư mục dự án.
2. Tạo và kích hoạt môi trường ảo theo mục 10.
3. Cài thư viện bằng `pip install -r requirements.txt`.
4. Khởi chạy Jupyter:

   ```bash
   jupyter notebook
   ```

5. Mở notebook `ex (14)(1).ipynb` (hoặc `main.ipynb` nếu đã đổi tên).
6. Chạy các cell từ trên xuống dưới bằng **Run All**.

Notebook tự sinh lại dữ liệu CSV trước khi đọc dữ liệu. Các cell huấn luyện và lưu checkpoint phải chạy thành công trước khi chạy phần Prediction.

## 12. Các điểm cần biết trước khi sử dụng kết quả

1. **Experiment 3 chưa có activation:** `DeepRegressionMLP` hiện chỉ nối các tầng Linear. Không nên diễn giải kết quả này như bằng chứng về tác động của độ sâu MLP phi tuyến.
2. **Nhận xét Markdown cũ trong notebook chưa đồng bộ:** phần nhận xét Experiment 3 có các số liệu không khớp với bảng output. Hãy ưu tiên bảng kết quả được in từ `df_exp1`, `df_exp2`, `df_exp3` và phần Prediction.
3. **Seed trong một đoạn nhận xét bị ghi nhầm:** mã sinh dữ liệu sử dụng Training = 42, Testing = 43, Validation = 44. Một đoạn Markdown trong notebook ghi Validation = 43 và Testing = 44; README này sử dụng thứ tự theo mã thực thi.
4. **Hình kiến trúc ở cell cuối chưa phản ánh cấu hình tốt nhất:** `models_info` đang gán cố định `hidden_dim = 10`, `num_layers = 1`, `val_loss = 0.012345` cho cả ba mô hình. Hình được lưu tên `best_models_horizontal.png` dù bố cục thực tế là ba biểu đồ xếp dọc. Không nên dùng hình này làm minh chứng cho cấu hình tốt nhất cho đến khi sửa code.
5. **Đánh giá Test trong các thí nghiệm:** notebook tính Test MSE/RMSE cho từng cấu hình trong quá trình khảo sát. Nếu cần quy trình đánh giá nghiêm ngặt hơn, nên chỉ dùng Validation để chọn cấu hình và chỉ đánh giá Test sau khi đã chốt mô hình.
6. **Tính tái lập:** seed giúp kiểm soát nhiều nguồn ngẫu nhiên, nhưng kết quả và thời gian vẫn có thể khác theo phiên bản thư viện, phần cứng và môi trường chạy.

## 13. Kết luận

Dự án hiện thực hóa pipeline dùng PyTorch MLP để xấp xỉ hàm mật độ Gaussian 2D, bao gồm sinh dữ liệu, tiền xử lý, huấn luyện, đánh giá và trực quan hóa. Ba thí nghiệm lần lượt khảo sát quy mô dữ liệu, độ rộng mạng và số tầng Linear ẩn. Khi báo cáo kết quả, cần nêu rõ giới hạn của Experiment 3, kích thước nhỏ của Validation/Test và việc notebook hiện tính chỉ số Test cho từng cấu hình.
