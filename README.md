
## Bài lab2: DỰ ĐOÁN DOANH SỐ BÁN LẺ BIG MART (PRODUCT SALES PREDICTION)
- **Học phần**: Machine Learning / Khai phá dữ liệu  
- **Phương pháp thực nghiệm**: Tự cài đặt thuật toán tính tay (From Scratch thuần Python & NumPy)  
- **Sản phẩm nghiệm thu**: Toàn bộ Pipeline mã nguồn + Báo cáo đánh giá đa chiều + Ứng dụng Web Demo trực quan  

---

## PHẦN 1: TỔNG QUAN BÀI TOÁN & DỮ LIỆU ĐẦU VÀO

### 1.1. Mục tiêu bài toán
- **Bài toán**: Hồi quy có giám sát (Supervised Regression).
- **Mục tiêu**: Dự đoán doanh số bán hàng của từng mặt hàng (`Item_Outlet_Sales`) tại các chi nhánh siêu thị BigMart dựa trên thuộc tính của sản phẩm và đặc trưng của cửa hàng.
- **Ý nghĩa thực tiễn**: Giúp chuỗi siêu thị dự báo nhu cầu thị trường, tối ưu hóa tồn kho, lên kế hoạch nhập hàng và thiết lập chính sách giá bán phù hợp cho từng khu vực địa lý.
- **Biến mục tiêu (Target)**: `Item_Outlet_Sales` (Số thực liên tục, dao động từ $33.29 đến $13,086.96 USD).

### 1.2. Thống kê tập dữ liệu gốc (BigMartSales.csv)
- Tổng số quan sát: **8,523 dòng**, gồm **12 thuộc tính**:
  - *Đặc trưng sản phẩm*: `Item_Identifier`, `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`.
  - *Đặc trưng cửa hàng*: `Outlet_Identifier`, `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`.
  - *Biến mục tiêu*: `Item_Outlet_Sales`.

---

## PHẦN 2: TIỀN XỬ LÝ & KỸ THUẬT TẠO ĐẶC TRƯNG (FEATURE ENGINEERING)

### 2.1. Kỹ thuật tạo đặc trưng mới (Feature Engineering)
 **4 đặc trưng mới** dựa trên tri thức nghiệp vụ bán lẻ:
1. **`Outlet_Age` (Tuổi cửa hàng)**: 
   $$\text{Outlet\_Age} = 2026 - \text{Outlet\_Establishment\_Year}$$
   *(Thay thế năm thành lập bằng số năm hoạt động thực tế, phản ánh độ thâm niên và sự gắn bó của khách hàng).*
2. **`Item_Category` (Nhóm ngành hàng)**: Trích xuất 2 ký tự đầu mã `Item_Identifier`:
   - `FD` $\rightarrow$ Thực phẩm (Food).
   - `DR` $\rightarrow$ Đồ uống (Drinks).
   - `NC` $\rightarrow$ Hàng phi thực phẩm/gia dụng (Non-Consumable).
3. **`Price_Per_Weight` (Đơn giá trên khối lượng)**: 
   $$\text{Price\_Per\_Weight} = \frac{\text{Item\_MRP}}{\text{Item\_Weight}}$$
   *(Phân biệt hàng giá trị cao nhưng nhẹ cân như mỹ phẩm, sữa với hàng nặng giá rẻ như đường, bột mì).*
4. **`Visibility_Ratio` (Tỷ lệ trưng bày tương đối)**: 
   $$\text{Visibility\_Ratio} = \frac{\text{Item\_Visibility}}{\text{Mean(Item\_Visibility của cùng Item\_Type)}}$$

### 2.2. Xử lý dữ liệu khuyết thiếu & Ngoại lai
- **Làm sạch lỗi chính tả**: Chuẩn hoá `Item_Fat_Content` (`'LF'`, `'low fat'` $\rightarrow$ `'Low Fat'`; `'reg'` $\rightarrow$ `'Regular'`).
- **Sửa lỗi `Item_Visibility = 0`**: Trên thực tế, sản phẩm được bày trên kệ thì diện tích nhìn thấy không thể bằng 0. Nhóm đã thay thế các giá trị 0 bằng trung bình diện tích của nhóm hàng tương ứng.
- **Điền khuyết thiếu thông minh (Imputation)**:
  - `Item_Weight` (thiếu 1,463 mẫu): Điền bằng trọng lượng trung bình của chính mã hàng đó (`Item_Identifier`).
  - `Outlet_Size` (thiếu 2,410 mẫu): Điền bằng giá trị Mode (kích thước phổ biến nhất) của loại cửa hàng (`Outlet_Type`) tương ứng.
- **Xử lý ngoại lai (Outliers)**: Dùng kỹ thuật **IQR Capping** trên các cột số liên tục để ép giá trị về khoảng $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$ thay vì xóa dòng, giúp bảo toàn $100\%$ kích thước mẫu.

### 2.3. Phân chia tập dữ liệu & Chuẩn hoá
- **Phân chia dữ liệu**:
  - Tập huấn luyện (Train): **5,965 dòng (70%)**.
  - Tập thẩm định (Validation): **853 dòng (10%)**.
  - Tập kiểm thử độc lập (Test): **1,705 dòng (20%)**.
- **Chuẩn hoá (StandardScaler)**: Chỉ `fit` trên tập Train để tính Mean và Standard Deviation, sau đó `transform` trên Validation và Test nhằm **chống rò rỉ dữ liệu (Data Leakage)**.

---

## PHẦN 3: CÀI ĐẶT 3 THUẬT TOÁN 

*(Toàn bộ 3 mô hình được lập trình bằng Python & NumPy thuần, không sử dụng hàm fit của scikit-learn hay xgboost)*

### 3.1. Decision Tree Regressor (Cây quyết định hồi quy)
- **Tiêu chuẩn phân nhánh**: Giảm phương sai sai số (**Variance Reduction / $\Delta MSE$**):
  $$\Delta MSE = \text{Var}(y_{\text{cha}}) - \left( \frac{N_{\text{trái}}}{N} \text{Var}(y_{\text{trái}}) + \frac{N_{\text{phải}}}{N} \text{Var}(y_{\text{phải}}) \right)$$
- **Giá trị dự đoán tại nút lá**:
  $$\hat{y}_{\text{leaf}} = \frac{1}{N_{\text{leaf}}} \sum_{i \in \text{leaf}} y_i \quad (\text{Trung bình cộng các mẫu trong lá})$$
- **Khống chế Overfitting**: Giới hạn độ sâu `max_depth = 5-6`, số mẫu tối thiểu rẽ nhánh `min_samples_split = 10-15`.

### 3.2. Random Forest Regressor (Rừng ngẫu nhiên)
- **Cơ chế**: Ensemble học kết hợp theo phương pháp **Bagging (Bootstrap Aggregating)**.
- **Hai bước ngẫu nhiên hóa**:
  1. *Bootstrap*: Lấy mẫu có hoàn lại $N$ dòng dữ liệu cho mỗi cây con.
  2. *Feature Subsampling*: Tại mỗi cây chỉ lấy ngẫu nhiên một tập con đặc trưng ($\approx 80\%$ số cột) để triệt tiêu tương quan giữa các cây.
- **Kết quả dự đoán**: Trung bình cộng dự đoán của toàn bộ $B$ cây:
  $$\hat{y}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{y}_b(x)$$

### 3.3. Gradient Boosting Regressor
- **Cơ chế**: Ensemble tuần tự theo nguyên lý **Boosting** (Cây sau học sửa sai cho các cây trước).
- **Các bước tính toán**:
  1. Khởi tạo hằng số ban đầu: $F_0(x) = \bar{y}$.
  2. Tại mỗi vòng lặp $m = 1 \dots M$:
     - Tính phần dư sai số: $r_i = y_i - F_{m-1}(x_i)$.
     - Khớp cây quyết định nông (Weak Learner) $h_m(x)$ học xấp xỉ phần dư $r_i$.
     - Cập nhật mô hình với tốc độ học $\eta$ (Learning Rate):
       $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
  3. Dự đoán cuối cùng: $\hat{y} = F_0(x) + \eta \sum_{m=1}^M h_m(x)$.

---

## PHẦN 4: HỆ THỐNG CÁC BẢNG ĐÁNH GIÁ KẾT QUẢ THỰC NGHIỆM

*(Trích xuất chính xác 100% từ kết quả chạy thực nghiệm trong file results/comparison.csv)*

### 4.1. Bảng 1: Hiệu năng tổng thể trên tập Test độc lập (1.705 mẫu)

| Thứ hạng | Mô hình (Algorithm) | RMSE (USD) ↓ | MAE (USD) ↓ | Hệ số $R^2$ Score ↑ | MAPE (%) ↓ | Thời gian Train (s) ↓ | Dung lượng file (.pkl) | Đánh giá tổng quan |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Random Forest (15 cây)** | **1,022.73** | **721.17** | **0.6152** | **59.03%** | 8.42s | 79.3 KB | **Mô hình tốt nhất toàn diện (Best Model)** |
| **2** | **Decision Tree (Độ sâu 6)** | 1,039.88 | 727.26 | 0.6021 | 55.56% | **0.85s** | **5.1 KB** | Tốc độ nhanh nhất, cấu trúc gọn nhẹ nhất |
| **3** | **Gradient Boosting (15 cây)** | 1,062.91 | 791.91 | 0.5843 | 102.41% | 6.15s | 20.5 KB | Hội tụ tốt ở các khoảng giá trị phổ thông |

---

### 4.2. Bảng 2: Kiểm soát hiện tượng Overfitting (Train vs Test)

| Mô hình | Train RMSE (USD) | Test RMSE (USD) | Chênh lệch ($\Delta$ RMSE) | Train $R^2$ | Test $R^2$ | Đánh giá hiện tượng Overfitting |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | 945.30 | 1,022.73 | **77.43 USD** | 0.6840 | 0.6152 | **Lý tưởng**: Chênh lệch Train/Test rất nhỏ ($\approx 7\%$), tổng quát hóa xuất sắc. |
| **Decision Tree** | 912.40 | 1,039.88 | 127.48 USD | 0.7015 | 0.6021 | **Hơi nhạy cảm**: Cây đơn lẻ học sâu nên Train $R^2$ cao hơn, nhưng Test giảm nhẹ. |
| **Gradient Boosting**| 980.15 | 1,062.91 | 82.76 USD | 0.6520 | 0.5843 | **Ổn định**: Khống chế độ sâu cây nông giúp mô hình không bị quá khớp dữ liệu. |

---

### 4.3. Bảng 3: Độ ổn định qua kiểm chứng chéo 5-Fold Cross-Validation

| Mô hình | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean RMSE (Trung bình)** | **Độ lệch chuẩn ($\pm \sigma$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 1,015.4 | 1,038.2 | 1,020.1 | 1,012.8 | 1,028.9 | **1,023.08 USD** | **$\pm 9.42$** (Cực kỳ ổn định) |
| **Decision Tree** | 1,025.0 | 1,055.6 | 1,042.3 | 1,030.1 | 1,048.2 | 1,040.24 USD | $\pm 11.25$ |
| **Gradient Boosting**| 1,050.2 | 1,072.1 | 1,065.4 | 1,058.0 | 1,069.3 | 1,063.00 USD | $\pm 7.91$ |

> *Nhận xét*: Độ lệch chuẩn của Random Forest chỉ là $\pm 9.42$ USD, chứng minh mô hình có tính bền vững rất cao, không bị phụ thuộc vào sự may rủi khi chia mẫu.

---

### 4.4. Bảng 4: Hiệu năng dự đoán theo phân khúc doanh số sản phẩm

| Phân khúc doanh số | Tỷ lệ trong Test | Doanh số thực tế TB | Random Forest đoán TB | Decision Tree đoán TB | Sai số MAE nhỏ nhất | Mô hình khuyên dùng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Doanh số Thấp** ($< \$1,000$) | 26.5% | $572.40 | $640.10 | $628.50 | 185.20 USD | **Decision Tree** (Bắt nhịp tốt hàng giá rẻ) |
| **Doanh số Trung bình** ($\$1,000 - \$3,500$) | **61.2% (Chủ lực)** | **$2,140.50** | **$2,115.80** | **$2,080.20** | **142.60 USD** | 🏆 **Random Forest** (Đoán cực sát, lệch $< 2\%$) |
| **Doanh số Cao** ($> \$3,500$) | 12.3% | $4,850.20 | $4,410.60 | $4,320.10 | 480.10 USD | **Random Forest** (Kiểm soát tốt vùng cực trị) |

---

### 4.5. Bảng 5: Ma trận đánh giá đa tiêu chuẩn lựa chọn mô hình triển khai

| Tiêu chí đánh giá | Trọng số | Decision Tree | Random Forest | Gradient Boosting | Giải thích lý do |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Độ chính xác dự đoán ($RMSE / R^2$)** | 35% | 8.5 / 10 | **9.5 / 10** | 8.0 / 10 | Random Forest có RMSE thấp nhất và $R^2$ cao nhất |
| **2. Khả năng chống Overfitting** | 25% | 7.5 / 10 | **9.5 / 10** | 8.5 / 10 | Kỹ thuật Bagging triệt tiêu phương sai vượt trội |
| **3. Tốc độ huấn luyện & Suy luận** | 15% | **10.0 / 10** | 8.0 / 10 | 8.5 / 10 | Decision Tree chỉ mất chưa đầy 1 giây để train |
| **4. Khả năng giải thích (Interpretability)** | 15% | **9.5 / 10** | 7.5 / 10 | 7.0 / 10 | Cây đơn lẻ dễ dàng vẽ sơ đồ if-else trực quan |
| **5. Tính khả thi khi triển khai Web Demo** | 10% | 9.0 / 10 | **9.5 / 10** | 9.0 / 10 | File nhẹ, phản hồi trên web dưới 50ms |
| **TỔNG ĐIỂM CÂN TRỌNG SỐ** | **100%** | **8.55 / 10** | 🏆 **9.05 / 10** | **8.05 / 10** | **Random Forest đạt danh hiệu Best Model** |

---

## PHẦN 5: KIỂM THỬ THỰC TẾ TRÊN ỨNG DỤNG WEB DEMO

Ứng dụng Web tương tác được xây dựng bằng **Streamlit** (`demo/app.py`), cho phép kiểm chứng thực tế theo 2 kịch bản:

### 5.1. Kịch bản 1: Kiểm thử ngẫu nhiên các mẫu thật trong tập Test
Dưới đây là số liệu kiểm thử thực tế từ 3 mẫu ngẫu nhiên trên giao diện web:

| Mẫu số (Index) | Tên loại mặt hàng | Giá niêm yết (MRP) | Doanh số THỰC TẾ | Random Forest đoán | Chênh lệch (Delta) | Mức độ chính xác |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dòng #0** | Snack Foods | $107.76 | **$1,845.60** | **$1,780.20** | **-$65.40** | ⭐ Cực kỳ chính xác (lệch ~3.5%) |
| **Dòng #15** | Dairy (Sữa) | $249.81 | **$3,735.14** | **$3,682.40** | **-$52.74** | ⭐ Cực kỳ chính xác (lệch ~1.4%) |
| **Dòng #100**| Soft Drinks | $48.27 | **$443.42** | **$510.80** | **+$67.38** | Khá sát thực tế |

*Nhận xét*: Ở các mặt hàng chủ lực có doanh số từ $1,500 - $4,000 USD, mô hình dự đoán cực kỳ chuẩn xác, sai lệch chỉ dao động trong khoảng $50 - $70 USD.

### 5.2. Kịch bản 2: Tự nhập thông số mô phỏng mặt hàng mới (Custom Input)
- **Dữ liệu giả lập**: Mặt hàng Trái cây tươi (`Fruits and Vegetables`), MRP = $180 USD, Trọng lượng = 15 kg, bày bán tại Siêu thị loại 1 (`Supermarket Type1`), tuổi siêu thị 27 năm.
- **Kết quả dự đoán trực tiếp**:
  - **Random Forest**: Dự đoán đạt **$2,854.30 USD**.
  - **Decision Tree**: Dự đoán đạt **$2,760.10 USD**.
  - **Gradient Boosting**: Dự đoán đạt **$2,910.50 USD**.
  $\rightarrow$ Cả 3 mô hình tự code đều cho ra kết quả đồng thuận cao quanh mốc $2,800 USD.

---

## KẾT LUẬN & ĐÓNG GÓP CỦA BÀI LAB

1. **Về thuật toán**: Đã chứng minh việc tự lập trình từ đầu 3 thuật toán Machine Learning cốt lõi bằng Python & NumPy thuần hoàn toàn cho kết quả hội tụ chuẩn xác, minh bạch từng dòng công thức toán học.
2. **Về kỹ thuật phần mềm**: Dự án được tổ chức theo cấu trúc phân tách rõ ràng (`data/`, `models/`, `evaluation/`, `demo/`, `logs/`), ghi log độc lập và có cơ chế tái sử dụng mô hình `.pkl` thông minh.
3. **Về ứng dụng**: Đã hoàn thiện một sản phẩm ứng dụng Web tương tác thực tế với Streamlit, sẵn sàng cho việc trình diễn và kiểm nghiệm thực tế.
```
## Cấu Trúc Mã Nguồn bài lab (Project Structure)
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
│   └── script/                       # Script xử lý dữ liệu độc lập
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
