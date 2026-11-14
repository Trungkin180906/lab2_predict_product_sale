import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "preprocessing.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def run_preprocessing(df: pd.DataFrame, target_col: str = "Item_Outlet_Sales"):
    df = df.copy()

    # 1.Sửa Fat Content gõ sai chính tả
    fat_map = {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    if "Item_Fat_Content" in df.columns:
        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(fat_map)

    # 2.Sửa Item_Visibility = 0
    if "Item_Visibility" in df.columns and "Item_Type" in df.columns:
        mean_vis = df.groupby("Item_Type")["Item_Visibility"].mean()
        mask_zero = df["Item_Visibility"] == 0
        df.loc[mask_zero, "Item_Visibility"] = df.loc[mask_zero, "Item_Type"].map(mean_vis)

    # 3.Điền khuyết Item_Weight và Outlet_Size
    if "Item_Weight" in df.columns:
        df["Item_Weight"] = df.groupby("Item_Identifier")["Item_Weight"].transform(lambda x: x.fillna(x.mean()))
        df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mean())

    if "Outlet_Size" in df.columns:
        mode_size = df.groupby("Outlet_Type")["Outlet_Size"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Medium")
        df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Type"].map(mode_size))

    # 4.Giới hạn ngoại lai IQR Capping cho các cột số
    for col in ["Item_Weight", "Item_Visibility", "Item_MRP"]:
        if col in df.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    # 5.Bỏ cột ID không dùng để train
    id_cols = [c for c in ["Item_Identifier", "Outlet_Identifier"] if c in df.columns]
    df.drop(columns=id_cols, inplace=True)

    # 6.Mã hoá Categorical
    for col in ["Outlet_Size", "Outlet_Location_Type"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    nom_cols = [c for c in ["Item_Fat_Content", "Item_Type", "Outlet_Type", "Item_Category"] if c in df.columns]
    df = pd.get_dummies(df, columns=nom_cols, drop_first=True, dtype=int)

    #Lưu cleaned data
    df.to_csv(PROCESSED_DIR / "cleaned_data.csv", index=False)

    # 7.Chia tập dữ liệu (70% Train, 10% Valid, 20% Test)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

    # 8.Chuẩn hoá StandardScaler (chỉ fit trên train)
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid[num_cols] = scaler.transform(X_valid[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
 
    # Lưu train/valid/test csv
    pd.concat([X_train, y_train], axis=1).to_csv(PROCESSED_DIR / "train.csv", index=False)
    pd.concat([X_valid, y_valid], axis=1).to_csv(PROCESSED_DIR / "valid.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_DIR / "test.csv", index=False)

    logger.info("Đã hoàn tất toàn bộ tiền xử lý và lưu data vào data/processed/.")
    return X_train, X_valid, X_test, y_train, y_valid, y_test, scaler
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "raw" / "BigMartSales.csv")
    run_preprocessing(df)