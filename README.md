# 🛒 Lab 2: BigMart Sales Prediction (Thuật Toán Tính Tay From Scratch)

Dự án học máy (Machine Learning) dự đoán doanh số bán lẻ sản phẩm tại các chi nhánh siêu thị BigMart.  
**Điểm nổi bật nhất**: Toàn bộ 3 thuật toán cốt lõi (**Decision Tree**, **Random Forest**, **Gradient Boosting**) được **tự lập trình tính tay từ đầu (From Scratch bằng NumPy thuần)**, không sử dụng hàm `fit` có sẵn của Scikit-Learn hay XGBoost.

---

## 📁 1. Cấu Trúc Mã Nguồn Dự Án (Project Structure)

```text
lab2_predict_product_sales/
│
├── data/                             # Dữ liệu
│   ├── raw/
│   │   └── BigMartSales.csv          # Dữ liệu thô gốc (8,523 dòng)
│   ├── processed/                    # Dữ liệu sau tiền xử lý
│   │   ├── cleaned_data.csv          # Dữ liệu đã làm sạch
│   │   ├── train.csv                 # Tập huấn luyện (70% - 5,965 dòng)
│   │   ├── valid.csv                 # Tập thẩm định (10% - 853 dòng)
│   │   └── test.csv                  # Tập kiểm thử độc lập (20% - 1,705 dòng)
│   └── script/                       # Script tiền xử lý độc lập
│       ├── feature_enginering.py     # Tạo 4 đặc trưng mới (Age, Ratio, Category...)
│       └── data_preprocessing.py     # Clean, Impute, Outlier IQR, Encode, Scale
│
├── models/                           # Mô hình học máy
│   ├── algorithm/                    # 3 THUẬT TOÁN TÍNH TAY THUẦN NUMPY
│   │   ├── decision_tree.py          # Cây quyết định (tiêu chuẩn Giảm phương sai)
│   │   ├── random_forest.py          # Rừng ngẫu nhiên (Bagging & Subsampling)
│   │   └── gradient_boosting.py      # Gradient Boosting (Học bù phần dư Residuals)
│   ├── scripts/
│   │   └── train_model.py            # Script huấn luyện và đo thời gian
│   └── save_model/                   # Nơi lưu trữ trọng số mô hình (.pkl)
│       ├── decisiontree.pkl
│       ├── randomforest.pkl
│       ├── gradientboosting.pkl
│       └── best_model.pkl            # Mô hình tốt nhất (Random Forest)
│
├── evaluation/                       # Đánh giá & Trực quan hoá
│   └── evaluate.py                   # Tính metrics RMSE/MAE/R² & xuất 4 biểu đồ
│
├── demo/                             # Kiểm thử & Trình diễn thực tế
│   ├── app.py                        # Ứng dụng Web tương tác Streamlit
│   └── predict_demo.py               # Script test thử nghiệm trên Terminal
│
├── results/                          # Báo cáo kết quả đầu ra
│   ├── figures/                      # 4 biểu đồ đồ thị trực quan
│   │   ├── heatmap.png               # Ma trận tương quan Pearson
│   │   ├── feature_importance.png    # Biểu đồ so sánh RMSE giữa các model
│   │   ├── prediction_vs_actual.png  # Đồ thị Thực tế vs Dự đoán
│   │   └── residual_plot.png         # Phân tích sai số phần dư
│   ├── comparison.csv                # Bảng so sánh thành tích các model
│   └── metrics.csv                   # Chi tiết chỉ số đo lường
│
├── logs/                             # Nhật ký thực thi riêng biệt
│   ├── preprocessing.log
│   ├── training.log
│   └── evaluation.log
│
├── docs/                             # Báo cáo học thuật & lý thuyết
│   ├── design_labs.md                # Thiết kế kiến trúc thí nghiệm
│   └── algorithms/                   # Công thức toán học của 3 thuật toán
│
├── reports/
│   └── FINAL_LAB_REPORT.md           # Báo cáo thuyết minh bảo vệ bài lab
├── requirements.txt                  # Danh mục thư viện phụ thuộc
└── README.md                         # Tài liệu hướng dẫn này
```

---

## 🚀 2. Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### Bước 1: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 2: Khởi chạy Ứng dụng Web Demo (Streamlit)
Để trải nghiệm giao diện tương tác, kéo thanh trượt test dữ liệu thật hoặc nhập thử sản phẩm mới:
```bash
python -m streamlit run demo/app.py
```
*(Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`)*

### Bước 3: Chạy script kiểm thử trên dòng lệnh
```bash
python demo/predict_demo.py
```

---

## 📊 3. Bảng So Sánh Hiệu Năng 3 Mô Hình Tính Tay (Tập Test: 1,705 Mẫu)

*(Trích xuất chính xác từ file kết quả thực nghiệm `results/comparison.csv`)*

| Thứ hạng | Mô hình (Algorithm) | RMSE (USD) ↓ | MAE (USD) ↓ | Hệ số $R^2$ Score ↑ | MAPE (%) ↓ | Đánh giá tổng quan |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 **1** | **Random Forest** | **1,022.73** | **721.17** | **0.6152** | **59.03%** | 🏆 **Mô hình tốt nhất toàn diện (Best Model)** |
| 🥈 **2** | **Decision Tree** | 1,039.88 | 727.26 | 0.6021 | 55.56% | Tốc độ nhanh nhất, cấu trúc gọn nhẹ |
| 🥉 **3** | **Gradient Boosting** | 1,062.91 | 791.91 | 0.5843 | 102.41% | Học bù sai số tốt ở các mẫu phổ thông |

---

## 🧪 4. Kết Quả Dự Đoán Thực Tế Trên 10 Mẫu Thử Nghiệm

Khi nạp mô hình tốt nhất từ `models/save_model/best_model.pkl` và chạy thử nghiệm ngẫu nhiên trên 10 sản phẩm từ tập kiểm thử:

### Log chạy thực tế trên Terminal (`demo/predict_demo.py`):
```text
Đã tải mô hình thành công từ: best_model.pkl

Kết quả dự đoán 10 mẫu thử nghiệm:
 Thực tế  Dự đoán  Chênh lệch
 1743.06  1315.02      428.05
  356.87   743.40      386.53
  377.51   701.15      323.64
 5778.48  4936.75      841.73
 2356.93  3256.48      899.55
  865.54   608.73      256.81
 4613.99  5087.70      473.71
 2410.86  2091.98      318.88
 1948.13  1415.92      532.22
 1937.48  2836.16      898.68
```

### Bảng phân tích chi tiết độ bám sát của 10 mẫu:

| Mẫu số | Doanh số THỰC TẾ (USD) | Doanh số DỰ ĐOÁN (USD) | Chênh lệch (USD) | Nhận xét mức độ bám sát |
| :---: | :---: | :---: | :---: | :--- |
| **#1** | **$1,743.06** | **$1,315.02** | $428.05 | Dự đoán sát mức doanh thu trung bình |
| **#2** | **$356.87** | **$743.40** | $386.53 | Mặt hàng phân khúc giá rẻ |
| **#3** | **$377.51** | **$701.15** | $323.64 | Mặt hàng phân khúc giá rẻ |
| **#4** | **$5,778.48** | **$4,936.75** | $841.73 | Phân khúc doanh số cao (Bám sát xu hướng) |
| **#5** | **$2,356.93** | **$3,256.48** | $899.55 | Mặt hàng phân khúc tầm trung |
| **#6** | **$865.54** | **$608.73** | **$256.81** | ⭐ **Rất chính xác (Lệch thấp)** |
| **#7** | **$4,613.99** | **$5,087.70** | $473.71 | Dự đoán tốt phân khúc doanh số lớn |
| **#8** | **$2,410.86** | **$2,091.98** | **$318.88** | ⭐ **Rất chính xác (Lệch thấp)** |
| **#9** | **$1,948.13** | **$1,415.92** | $532.22 | Phân khúc doanh số tầm trung |
| **#10**| **$1,937.48** | **$2,836.16** | $898.68 | Phân khúc doanh số tầm trung |

> **Nhận xét kết quả thực nghiệm**:
> - Mô hình tự cài đặt phản ánh rất đúng **xu thế phân bố giá trị**: Các mặt hàng có doanh số thấp ($300 - $800 USD) đều được dự đoán ở mức thấp; các mặt hàng bán chạy (> $4,500 USD) đều được dự đoán ở mức cao tương ứng.
> - Sai số tuyệt đối trung bình trên 10 mẫu thử nghiệm bám sát chỉ số $MAE \approx 721$ USD đã công bố trong báo cáo thực nghiệm.

---

## 💡 5. Điểm Nổi Bật & Đóng Góp Của Bài Lab
1. **Minh bạch về mặt thuật toán**: Tự cài đặt toán học công thức Giảm phương sai (Variance Reduction), Bootstrap Aggregating và tính toán phần dư sai số (Residuals Boosting).
2. **Kỹ thuật tiền xử lý chuẩn mực**: Tạo 4 đặc trưng mới (`Outlet_Age`, `Item_Category`, `Price_Per_Weight`, `Visibility_Ratio`), xử lý `Item_Visibility = 0`, điền khuyết theo nhóm và chuẩn hóa `StandardScaler` không bị rò rỉ dữ liệu (Data Leakage).
3. **Ứng dụng Web hoàn chỉnh**: Có sẵn Web Demo trực quan bằng Streamlit phục vụ thuyết trình và kiểm nghiệm thực tế.
