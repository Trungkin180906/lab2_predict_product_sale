import sys
from pathlib import Path

#Thêm thư mục gốc vào PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import pandas as pd

def run_demo():
    model_path = ROOT/"models"/"save_model"/"best_model.pkl"
    test_path = ROOT/"data"/"processed"/"test.csv"

    if not model_path.exists():
        print("Chưa có file models/save_model/best_model.pkl! Hãy chạy 'python main.py' trước.")
        return

    # Tải mô hình tự viết
    model = joblib.load(model_path)
    print(f"Đã tải mô hình thành công từ: {model_path.name}")

    # Đọc dữ liệu test để lấy 5 mẫu demo
    test_df = pd.read_csv(test_path)
    samples = test_df.drop(columns=["Item_Outlet_Sales"]).head(10)
    actuals = test_df["Item_Outlet_Sales"].head(10).values

    # Dự đoán bằng hàm predict tính tay
    predictions = model.predict(samples)

    result_df = pd.DataFrame({
        "Thực tế": [round(a, 2) for a in actuals],
        "Dự đoán": [round(p, 2) for p in predictions],
        "Chênh lệch": [round(abs(a - p), 2) for a, p in zip(actuals, predictions)]
    })

    print("\nKết quả dự đoán 10 mẫu thử nghiệm:")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    run_demo()