import logging
from pathlib import Path
import pandas as pd

#Tự cấu hình log ra logs/preprocessing.log
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "preprocessing.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def run_feature_engineering(df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    df = df.copy()

    # 1.Tuổi cửa hàng (Outlet_Age)
    if "Outlet_Establishment_Year" in df.columns:
        df["Outlet_Age"] = current_year - df["Outlet_Establishment_Year"]
        df.drop(columns=["Outlet_Establishment_Year"], inplace=True)
        logger.info("Đã tạo đặc trưng: Outlet_Age")

    # 2.Nhóm ngành hàng từ 2 ký tự đầu (FD=Food, DR=Drinks, NC=Non-Consumable)
    if "Item_Identifier" in df.columns:
        cat_map = {"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"}
        df["Item_Category"] = df["Item_Identifier"].str[:2].map(cat_map).fillna("Other")
        logger.info("Đã tạo đặc trưng: Item_Category")

    # 3.Đơn giá trên khối lượng (Price_Per_Weight)
    if "Item_MRP" in df.columns and "Item_Weight" in df.columns:
        safe_w = df["Item_Weight"].replace(0, 1)
        df["Price_Per_Weight"] = df["Item_MRP"] / safe_w
        logger.info("Đã tạo đặc trưng: Price_Per_Weight")

    # 4.Tỷ lệ diện tích trưng bày so với nhóm hàng (Visibility_Ratio)
    if "Item_Visibility" in df.columns and "Item_Type" in df.columns:
        mean_vis = df.groupby("Item_Type")["Item_Visibility"].transform("mean").replace(0, 0.001)
        df["Visibility_Ratio"] = df["Item_Visibility"] / mean_vis
        logger.info("Đã tạo đặc trưng: Visibility_Ratio")
    return df