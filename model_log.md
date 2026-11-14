# NHẬT KÝ THÔNG SỐ VÀ CẤU HÌNH MÔ HÌNH (MODEL CONFIG & LOG)
> **Dự án:** Dự đoán doanh số bán hàng BigMart Sales (Lab 2)  
> **Thời gian xuất log:** `2026-11-15 00:22:01`  
> **Môi trường thực thi:** Python 3.12 (Windows) | Thuật toán tự xây dựng từ đầu (From Scratch with NumPy)

---

## 1. THÔNG TIN TẬP DỮ LIỆU ĐẦU VÀO
* **Tổng số mẫu huấn luyện (Train samples):** `5,965` dòng
* **Số chiều đặc trưng (Features):** `29` đặc trưng
* **Phân chia dữ liệu:** `70% Train` | `10% Validation` | `20% Test`
* **Danh sách đặc trưng đầu vào (29 features):**
  > `Item_Weight, Item_Visibility, Item_MRP, Outlet_Size, Outlet_Location_Type, Outlet_Age, Price_Per_Weight, Visibility_Ratio, Item_Fat_Content_Regular, Item_Type_Breads, Item_Type_Breakfast, Item_Type_Canned, Item_Type_Dairy, Item_Type_Frozen Foods, Item_Type_Fruits and Vegetables, Item_Type_Hard Drinks, Item_Type_Health and Hygiene, Item_Type_Household, Item_Type_Meat, Item_Type_Others, Item_Type_Seafood, Item_Type_Snack Foods, Item_Type_Soft Drinks, Item_Type_Starchy Foods, Outlet_Type_Supermarket Type1, Outlet_Type_Supermarket Type2, Outlet_Type_Supermarket Type3, Item_Category_Food, Item_Category_Non-Consumable`

---

## 2. BẢNG TỔNG HỢP SIÊU THAM SỐ (HYPERPARAMETERS)
*Các tham số cấu hình do người lập trình thiết lập trước khi huấn luyện mô hình:*

| Thuật toán | Cơ chế thuật toán | Siêu tham số (Hyperparameters) | Giá trị cấu hình | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- | :---: | :--- |
| **Decision Tree** | Variance Reduction (MSE) | `max_depth`<br>`min_samples_leaf` | **6**<br>**5** | Khống chế độ sâu cây để chống Overfitting; Dừng rẽ nhánh nếu lá có ít hơn 5 mẫu. |
| **Random Forest** | Bagging + Subsampling | `n_estimators`<br>`max_depth`<br>`min_samples_leaf`<br>`max_features` | **15 cây**<br>**6**<br>**5**<br>**"sqrt"** | Số cây trong rừng; Chọn ngẫu nhiên căn bậc 2 số đặc trưng tại mỗi node để giảm tương quan giữa các cây. |
| **Gradient Boosting**| Sequential Residual Boosting| `n_estimators`<br>`learning_rate`<br>`max_depth`<br>`min_samples_leaf` | **15 cây**<br>**0.1**<br>**4**<br>**5** | Số vòng lặp sửa sai; Tốc độ học $\eta = 0.1$ giúp mô hình hội tụ từ từ, tránh phân kỳ hàm mất mát. |

---

## 3. THAM SỐ HỌC ĐƯỢC CỦA MÔ HÌNH (LEARNED PARAMETERS)
*Các giá trị mà mô hình tự động học và lưu trữ vào file nhị phân sau khi khớp dữ liệu:*

* **Decision Tree (`decisiontree.pkl`):**
  * Node gốc rẽ nhánh theo đặc trưng index: `#2` với ngưỡng giá trị: `0.0405`.
* **Random Forest (`randomforest.pkl`):**
  * Số lượng cây quyết định thực tế đã khởi tạo và hội tụ: **15 cây**.
* **Gradient Boosting (`gradientboosting.pkl`):**
  * Giá trị dự đoán khởi tạo ban đầu ($y_0 = \text{mean}(y)$): **$0.00 USD**.
  * Số lượng cây học phần dư (Residual Trees) đã hoàn thành: **15 cây**.

---

## 4. KẾT QUẢ HIỆU NĂNG TRÊN TẬP KIỂM THỬ (TEST SET)

| Model            |    RMSE |    MAE |     R2 |   MAPE |
|:-----------------|--------:|-------:|-------:|-------:|
| RandomForest     | 1022.73 | 721.17 | 0.6152 |  59.03 |
| DecisionTree     | 1039.88 | 727.26 | 0.6021 |  55.56 |
| GradientBoosting | 1062.91 | 791.91 | 0.5843 | 102.41 |

###  Kết luận lựa chọn mô hình:
* **Mô hình tốt nhất (Best Model):** `Random Forest` đạt hiệu năng dẫn đầu với **$R^2 = 0.6152$** và sai số tuyệt đối thấp nhất **$MAE = 721.17 USD$**.
* Mô hình được chọn làm chuẩn mặc định cho hệ thống **Production Deployment** trên ứng dụng Web Streamlit (`demo/app.py`).
