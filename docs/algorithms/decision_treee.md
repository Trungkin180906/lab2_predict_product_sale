
#  THUẬT TOÁN DECISION TREE REGRESSOR 

## 1. Ý tưởng toán học
Cây quyết định hồi quy phân chia không gian dữ liệu nhiều chiều thành các vùng không gian hình chữ nhật rời rạc. Mỗi bước phân chia (split) nhằm mục đích làm cho các giá trị $y$ trong từng vùng con đồng nhất nhất có thể.

## 2. Tiêu chí phân chia: Giảm phương sai (Variance Reduction)
Tại mỗi nút cha $P$, thuật toán tìm kiếm cặp thuộc tính $j$ và ngưỡng $s$ (threshold) sao cho:
$$\Delta MSE = \text{Var}(y_P) - \left[ \frac{N_L}{N} \text{Var}(y_L) + \frac{N_R}{N} \text{Var}(y_R) \right]$$

Trong đó:
- $\text{Var}(y) = \frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2$ là phương sai (tương đương với Mean Squared Error tại nút).
- $N_L, N_R$ là số lượng mẫu thuộc nhánh con bên trái ($x_j \le s$) và bên phải ($x_j > s$).
- Mục tiêu: Chọn cặp $(j, s)$ sao cho $\Delta MSE$ đạt giá trị lớn nhất.

## 3. Dự đoán tại nút lá (Leaf Value)
Khi đạt điều kiện dừng (độ sâu tối đa `max_depth` hoặc số mẫu tối thiểu `min_samples_split`), nút trở thành nút lá:
$$\hat{y}_{leaf} = \frac{1}{N_{leaf}} \sum_{i \in leaf} y_i$$
Giá trị dự đoán là **trung bình cộng** của tất cả các mẫu rơi vào nút lá đó.

## 4. Tối ưu hoá tốc độ tính toán bằng NumPy
- Thay vì duyệt qua hàng ngàn giá trị thực của mỗi cột, thuật toán chia mốc phân vị (Quantiles từ 10% đến 90%) để giảm số phép tính từ $O(N \times D)$ xuống mức cố định, giúp cây tính tay fit cực nhanh.
