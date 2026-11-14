# THUẬT TOÁN RANDOM FOREST REGRESSOR

## 1. Cơ chế hoạt động: Ensemble Bagging
Một cây quyết định đơn lẻ thường có phương sai rất lớn (dễ bị Overfitting trên tập Train). Random Forest khắc phục nhược điểm này bằng cách kết hợp dự đoán của $B$ cây quyết định độc lập theo phương pháp **Bootstrap Aggregating (Bagging)**.

## 2. Hai kỹ thuật ngẫu nhiên cốt lõi

### 2.1. Lấy mẫu có hoàn lại (Bootstrap Sampling)
- Cho tập dữ liệu gốc gồm $N$ mẫu.
- Với mỗi cây thứ $b$ ($b = 1, \dots, B$), chọn ngẫu nhiên $N$ mẫu từ tập gốc **có lặp lại** (`replace=True`).
- Về mặt lý thuyết xác suất, khoảng $63.2\%$ mẫu độc nhất sẽ được chọn vào tập huấn luyện của mỗi cây, phần còn lại ($36.8\%$) đóng vai trò kiểm thử ngoại kiểm (Out-Of-Bag).

### 2.2. Lựa chọn tập con đặc trưng (Feature Subsampling)
- Khi xây dựng mỗi cây, thuật toán không dùng toàn bộ $D$ đặc trưng mà chỉ chọn ngẫu nhiên một tập con gồm:
  $$d = \text{int}(D \times \text{max\_features\_ratio})$$
- Việc này giúp phá vỡ sự tương quan (Decorrelate) giữa các cây con trong rừng, đảm bảo khi một đặc trưng quá mạnh cũng không chi phối toàn bộ các cây.

## 3. Công thức dự đoán tổng hợp
Với một mẫu dữ liệu mới $x$, mỗi cây $b$ đưa ra một dự đoán độc lập $\hat{y}_b(x)$. Kết quả cuối cùng là **trung bình cộng** của toàn bộ khu rừng:
$$\hat{y}_{RF}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{y}_b(x)$$

*Hiệu quả*: Giảm thiểu phương sai của mô hình xuống gần $1/B$ lần mà không làm tăng độ lệch (Bias).
