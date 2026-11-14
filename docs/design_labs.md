# BÁO CÁO THIẾT KẾ BÀI LAB 2: DỰ ĐOÁN DOANH SỐ BÁN LẺ (BIGMART SALES)

## 1. TỔNG QUAN BÀI TOÁN
- **Mục tiêu**: Xây dựng mô hình học máy có giám sát (Supervised Regression) nhằm dự đoán chính xác doanh số bán hàng của từng mặt hàng (`Item_Outlet_Sales`) tại các chi nhánh cửa hàng siêu thị BigMart.
- **Ý nghĩa thực tiễn**: Giúp chuỗi bán lẻ dự báo nhu cầu thị trường, tối ưu hóa tồn kho, lên kế hoạch nhập hàng và thiết lập chính sách giá bán phù hợp cho từng khu vực địa lý.
- **Biến mục tiêu (Target)**: `Item_Outlet_Sales` (Số thực liên tục - Doanh số bán ra).

---

## 2. KIẾN TRÚC TOÀN BỘ HỆ THỐNG (END-TO-END PIPELINE)

```text
Dữ liệu thô (BigMartSales.csv)
       │
       ▼
[1] Feature Engineering (Tạo đặc trưng mới: Age, Category, Price/Weight, Visibility Ratio)
       │
       ▼
[2] Data Preprocessing
   ├── Làm sạch lỗi chính tả (Item_Fat_Content: LF, reg...)
   ├── Xử lý giá trị phi lý (Item_Visibility = 0)
   ├── Điền khuyết thiếu (Item_Weight theo mã ID, Outlet_Size theo Mode loại cửa hàng)
   ├── Ép ngưỡng ngoại lai (IQR Capping trên Weight, Visibility, MRP)
   ├── Loại bỏ các cột ID không có ý nghĩa tính toán
   ├── Mã hoá: Label Encoding (Thứ tự) + One-Hot Encoding (Danh mục)
   ├── Chia dữ liệu: Train (70%) - Validation (10%) - Test (20%)
   └── Co giãn đặc trưng số: StandardScaler (Fit độc quyền trên Train)
       │
       ▼
[3] Model Training (3 Thuật toán TÍNH TAY thuần NumPy)
   ├── Decision Tree Regressor (from scratch)
   ├── Random Forest Regressor (from scratch)
   └── Gradient Boosting Regressor (from scratch)
       │
       ▼
[4] Model Evaluation & Comparison
   ├── Tính toán độ đo: RMSE, MAE, R², MAPE
   └── Xuất 4 biểu đồ báo cáo: Heatmap, Feature/Model Comparison, Pred vs Actual, Residual
       │
       ▼
[5] Live Demo Web Application (Streamlit Web UI)
