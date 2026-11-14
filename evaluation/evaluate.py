# evaluation/evaluate.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = ROOT/"results"/"figures"
RES_DIR = ROOT/"results"
LOG_DIR = ROOT/"logs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("evaluate")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "evaluation.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def calculate_metrics(y_true, y_pred, model_name: str):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        "Model": model_name,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2)
    }

def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        m = calculate_metrics(y_test, preds, name)
        rows.append(m)

    df_comp = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    df_comp.to_csv(RES_DIR / "comparison.csv", index=False)
    df_comp.to_csv(RES_DIR / "metrics.csv", index=False)
    logger.info(f"\n{df_comp.to_string(index=False)}")
    return df_comp

def plot_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Ma tran tuong quan Pearson", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "heatmap.png", dpi=150)
    plt.close()

def plot_model_comparison_bar(df_compare: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_compare["Model"], df_compare["RMSE"], color=["#1f77b4", "#2ca02c", "#d62728"])
    plt.title("So sanh sai so RMSE giua cac mo hinh tinh tay", fontsize=13, fontweight="bold")
    plt.ylabel("RMSE (Thap hon la tot hon)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.1f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close()

def plot_pred_vs_actual(y_true, y_pred, model_name: str):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.4, color="#1f77b4", edgecolor="k", s=25)
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([0, max_v], [0, max_v], "r--", linewidth=2, label="Duong hoan hao")
    plt.title(f"Du doan vs Thuc te ({model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Thuc te (Item_Outlet_Sales)")
    plt.ylabel("Du doan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prediction_vs_actual.png", dpi=150)
    plt.close()

def plot_residuals(y_true, y_pred, model_name: str):
    res = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, res, alpha=0.4, color="#e377c2", edgecolor="k", s=20)
    axes[0].axhline(y=0, color="red", linestyle="--")
    axes[0].set_title("Bieu do phan tan phan du", fontweight="bold")
    axes[0].set_xlabel("Du doan")
    axes[0].set_ylabel("Phan du")

    sns.histplot(res, kde=True, bins=40, ax=axes[1], color="#9467bd")
    axes[1].axvline(x=0, color="red", linestyle="--")
    axes[1].set_title("Phan phoi sai so", fontweight="bold")
    axes[1].set_xlabel("Phan du")

    plt.suptitle(f"Phan tich sai so phan du - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_plot.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    test_path = ROOT/"data"/"processed"/"test.csv"
    if not test_path.exists():
        print("Chua co file test.csv! Hay chay preprocessing truoc.")
        sys.exit(1)

    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=["Item_Outlet_Sales"])
    y_test = df_test["Item_Outlet_Sales"]

    save_dir = ROOT/"models"/"save_model"

    # Sửa đúng tên file: decisiontree.pkl, randomforest.pkl, gradientboosting.pkl
    models = {
        "DecisionTree": joblib.load(save_dir/"decisiontree.pkl"),
        "RandomForest": joblib.load(save_dir/"randomforest.pkl"),
        "GradientBoosting": joblib.load(save_dir/"gradientboosting.pkl")
    }

    # Đánh giá & So sánh
    print("-> Dang so sanh 3 mo hinh tren tap Test...")
    comparison = compare_models(models, X_test, y_test)
    print("\nBang so sanh ket qua:")
    print(comparison.to_string(index=False))

    # Xuất 4 biểu đồ báo cáo
    print("-> Dang ve 4 bieu do vao results/figures/...")
    plot_heatmap(df_test)
    plot_model_comparison_bar(comparison)

    best_name = comparison.iloc[0]["Model"]
    best_model = models[best_name]
    y_pred = best_model.predict(X_test)

    plot_pred_vs_actual(y_test, y_pred, best_name)
    plot_residuals(y_test, y_pred, best_name)

    # Lưu thêm 1 bản best_model.pkl
    joblib.dump(best_model, save_dir/"best_model.pkl")

    print(f"\n Danh gia hoan tat! Mo hinh tot nhat la: {best_name}")
    print(f"Cac bieu do da duoc luu tai: results/figures/")