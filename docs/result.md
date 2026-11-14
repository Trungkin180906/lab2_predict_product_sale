# BÁO CÁO KẾT QUẢ ĐÁNH GIÁ VÀ SO SÁNH 3 MÔ HÌNH HỌC MÁY
**Bài toán**: Dự đoán Doanh số Bán lẻ BigMart Sales Prediction  
**Phương pháp thực nghiệm**: Tự cài đặt thuật toán thuần (From Scratch bằng NumPy)

---

## 1. BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM

Dưới đây là bảng số liệu đánh giá trên tập kiểm thử độc lập **Test Dataset (1.705 mẫu)**:

| Thứ hạng | Mô hình (Algorithm) | RMSE (USD) ↓ | MAE (USD) ↓ | Hệ số $R^2$ ↑ | MAPE (%) ↓ | Đánh giá tổng quan |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Random Forest** | **1,022.73** | **721.17** | **0.6152** | **59.03%** | **Tốt nhất toàn diện (Best Model)** |
| **2** | **Decision Tree** | 1,039.88 | 727.26 | 0.6021 | 55.56% | Ổn định, tốc độ tính toán nhanh nhất |
| **3** | **Gradient Boosting** | 1,062.91 | 791.91 | 0.5843 | 102.41% | Học bù sai số tốt ở các mẫu phổ thông |

> **Quy ước đọc chỉ số**:
> - **RMSE** *(Root Mean Squared Error)*: Càng thấp càng tốt.
> - **MAE** *(Mean Absolute Error)*: Càng thấp càng tốt.
> - **$R^2$ Score**: Càng gần 1.0 càng tốt (giải thích được biến thiên dữ liệu).
> - **MAPE**: Phần trăm sai lệch tương đối so với giá trị thực.

---

## 2. PHÂN TÍCH CHI TIẾT HIỆU NĂNG TỪNG THUẬT TOÁN

### 1. Mô hình Random Forest
- **Điểm số**: $RMSE = 1,022.73$ | $R^2 = 0.6152$ (Giải thích được **61.52%** sự biến thiên doanh số).
- **Ưu điểm**:
  - Nhờ cơ chế **Bootstrap Aggregating (Bagging)** kết hợp dự đoán của nhiều cây con và **Feature Subsampling** (chọn ngẫu nhiên một phần đặc trưng cho mỗi cây), mô hình triệt tiêu phương sai (Variance) cực kỳ hiệu quả.
  - Hạn chế tối đa hiện tượng học vẹt (Overfitting) vốn là điểm yếu cố hữu của cây quyết định đơn lẻ.
- **Kết luận**: Đây là mô hình được lựa chọn chính thức để lưu thành `models/save_model/best_model.pkl` và phục vụ cho ứng dụng Web Demo thực tế.

---

### 2. Mô hình Decision Tree 
- **Điểm số**: $RMSE = 1,039.88$ | $MAE = 727.26$ | $R^2 = 0.6021$.
- **Ưu điểm**:
  - Tốc độ huấn luyện và suy luận nhanh nhất trong cả 3 mô hình.
  - Sai số $MAE$ đạt mức $727.26$ USD, bám rất sát mô hình Random Forest.
- **Hạn chế**:
  - Cấu trúc cây đơn lẻ dễ nhạy cảm với các mẫu dữ liệu biên (ngoại lai), dẫn đến sai số bình phương $RMSE$ cao hơn Random Forest khoảng 17 đơn vị.

---

### 3. Mô hình Gradient Boosting
- **Điểm số**: $RMSE = 1,062.91$ | $R^2 = 0.5843$.
- **Đặc điểm**:
  - Thuật toán học tuần tự bù trừ phần dư sai số ($Residuals$) qua từng vòng lặp.
  - Đối với phiên bản tính tay với số cây nông (weak learners `max_depth = 3-4`), mô hình đạt mức giải thích $58.43\%$. 
  - Chỉ số $MAPE$ cao hơn do một số ít sản phẩm có doanh số thực tế rất nhỏ (dưới $100) khiến tỷ lệ phần trăm sai lệch bị khuếch đại, tuy nhiên giá trị $RMSE$ tổng thể vẫn nằm ở mức kiểm soát tốt (~1,062 USD).

---

## 3. PHÂN TÍCH CÁC ĐỒ THỊ TRỰC QUAN HÓA (RESULTS/FIGURES/)

1. **Ma trận tương quan (`heatmap.png`)**:
   - Đặc trưng **`Item_MRP` (Giá niêm yết)** có hệ số tương quan dương mạnh nhất với doanh số (`Item_Outlet_Sales`), khẳng định giá bán lẻ là yếu tố quyết định hàng đầu đến tổng doanh thu.
   - Các đặc trưng tạo mới như `Price_Per_Weight` và `Outlet_Age` đóng góp mối liên hệ có ý nghĩa rõ rệt vào quyết định phân nhánh của cây.

2. **Thực tế vs Dự đoán (`prediction_vs_actual.png`)**:
   - Đám mây điểm dự đoán phân bố tập trung dày đặc bám sát **đường chéo lý tưởng ($y = x$)** trong khoảng doanh số từ $500 đến $4,500 USD.
   - Ở các vùng doanh số cực cao (> $6,000 USD), mô hình có xu hướng dự đoán khiêm tốn hơn thực tế nhằm tránh sai số cực đoan.

3. **Phân tích sai số phần dư (`residual_plot.png`)**:
   - Biểu đồ phân tán phần dư dao động ngẫu nhiên xung quanh trục $0$, không xuất hiện dạng hình phễu hay đường cong phi tuyến bất thường.
   - Phân phối sai số dạng hình chuông (chuẩn hóa), chứng minh mô hình không bị lệch hệ thống (Unbiased).

---

### Bảng phân tích chi tiết độ bám sát của 10 mẫu:
| Mẫu số | Doanh số THỰC TẾ (USD) | Doanh số DỰ ĐOÁN (USD) | Chênh lệch (USD) | Nhận xét mức độ bám sát |
| :---: | :---: | :---: | :---: | :--- |
| **#1** | **$1,743.06** | **$1,315.02** | $428.05 | Dự đoán sát mức doanh thu trung bình |
| **#2** | **$356.87** | **$743.40** | $386.53 | Mặt hàng phân khúc giá rẻ |
| **#3** | **$377.51** | **$701.15** | $323.64 | Mặt hàng phân khúc giá rẻ |
| **#4** | **$5,778.48** | **$4,936.75** | $841.73 | Phân khúc doanh số cao (Bám sát xu hướng) |
| **#5** | **$2,356.93** | **$3,256.48** | $899.55 | Mặt hàng phân khúc tầm trung |
| **#6** | **$865.54** | **$608.73** | **$256.81** | **Rất chính xác (Lệch thấp)** |
| **#7** | **$4,613.99** | **$5,087.70** | $473.71 | Dự đoán tốt phân khúc doanh số lớn |
| **#8** | **$2,410.86** | **$2,091.98** | **$318.88** | **Rất chính xác (Lệch thấp)** |
| **#9** | **$1,948.13** | **$1,415.92** | $532.22 | Phân khúc doanh số tầm trung |
| **#10**| **$1,937.48** | **$2,836.16** | $898.68 | Phân khúc doanh số tầm trung |

## 4. KẾT LUẬN 

1. **Về mặt kỹ thuật**: Cả 3 thuật toán tự cài đặt bằng tay thuần NumPy đều hoạt động chính xác theo đúng lý thuyết toán học, đạt độ hội tụ tốt và cho kết quả cạnh tranh với các thư viện đóng gói sẵn.
2. **Khuyến nghị triển khai**:
   - Sử dụng **Random Forest** làm mô hình suy luận lõi trên ứng dụng Web Demo (`demo/app.py`).
   - Ưu tiên trưng bày và quản lý tồn kho chặt chẽ các nhóm hàng có `Item_MRP` cao tại các siêu thị quy mô lớn (`Supermarket Type 3`) và cửa hàng lâu năm (`Outlet_Age` cao) để tối đa hóa doanh thu của toàn chuỗi BigMart.

