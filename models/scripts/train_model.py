import time
import logging
from pathlib import Path
import joblib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
#Import trực tiếp các thuật toán tự viết từ models/algorithm/
from models.algorithm.decision_tree import DecisionTreeRegressorScratch
from models.algorithm.random_forest import RandomForestRegressorScratch
from models.algorithm.gradient_boosting import GradientBoostingRegressorScratch

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT/"logs"
SAVE_DIR = ROOT /"models"/"save_model"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("train_models")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "training.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def train_all_scratch_models(X_train, y_train):
    logger.info("BẮT ĐẦU HUẤN LUYỆN 3 THUẬT TOÁN")

    models = {
        "DecisionTree": DecisionTreeRegressorScratch(max_depth=6, min_samples_split=10),
        "RandomForest": RandomForestRegressorScratch(n_estimators=15, max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingRegressorScratch(n_estimators=15, learning_rate=0.1, max_depth=4)
    }
    trained = {}
    for name, model in models.items():
        print(f"Đang huấn luyện: {name}...")
        start = time.time()
        #Fit bằng thuật toán tự code từ đầu
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        #Lưu file .pkl vào models/save_model/
        save_path = SAVE_DIR / f"{name.lower()}.pkl"
        joblib.dump(model, save_path)

        msg = f"Huấn luyện thành công {name} trong {elapsed:.2f}s -> Đã lưu: {save_path.name}"
        print(msg)
        logger.info(msg)
        trained[name] = model
    return trained
if __name__ == "__main__":
    import pandas as pd
    # Đọc dữ liệu train
    train_path = ROOT/"data"/"processed"/"train.csv"
    train_df = pd.read_csv(train_path)
    # Tách X và y
    X_train = train_df.drop(columns=["Item_Outlet_Sales"])
    y_train = train_df["Item_Outlet_Sales"]
    # Huấn luyện tất cả mô hình
    train_all_scratch_models(X_train, y_train)