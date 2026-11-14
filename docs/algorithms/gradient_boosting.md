# THUẬT TOÁN GRADIENT BOOSTING REGRESSOR 

## 1. Nguyên lý Boosting: Học tuần tự sửa sai
Khác với Random Forest (các cây học song song và độc lập), Gradient Boosting xây dựng các cây con **nối tiếp nhau**:
- Mỗi cây mới được tạo ra nhằm mục đích **học và bù đắp sai số (Residual)** của toàn bộ các cây phía trước.

## 2. Quy trình thuật toán toán học từng bước

### Bước 1: Khởi tạo mô hình cơ sở $F_0(x)$
Dự đoán khởi đầu là một hằng số tối ưu hóa hàm mất mát bình phương:
$$F_0(x) = \bar{y} = \frac{1}{N} \sum_{i=1}^{N} y_i$$

### Bước 2: Lặp qua từng bước boosting $m = 1, 2, \dots, M$
1. **Tính toán phần dư sai số (Pseudo-Residuals)** cho từng mẫu $i$:
   $$r_{im} = y_i - F_{m-1}(x_i)$$
   *(Với hàm mất mát sai số bình phương $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$, phần dư này chính là đạo hàm riêng âm $-\frac{\partial L}{\partial F}$)*.

2. **Khớp một cây quyết định nông (Weak Learner)** $h_m(x)$ để dự đoán tập phần dư $r_{im}$.

3. **Cập nhật mô hình** với tham số tốc độ học $\eta$ (Learning Rate):
   $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

### Bước 3: Dự đoán cuối cùng
Sau $M$ vòng lặp, mô hình hoàn chỉnh là tổng tích lũy:
$$F(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)$$

## 3. Vai trò của Tốc độ học ($\eta$ - Learning Rate)
- Tham số $\eta \in (0, 1]$ (thường chọn $0.05 - 0.1$) đóng vai trò thu nhỏ bước nhảy, giúp mô hình tiến từ từ về điểm tối ưu mà không bị Overfitting quá nhanh.
