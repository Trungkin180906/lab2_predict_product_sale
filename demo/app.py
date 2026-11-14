"""Giao diện Demo Streamlit – Dự đoán doanh số BigMart Sales."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAVE_DIR  = ROOT / "models" / "save_model"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"

MODEL_FILES = {
    "Decision Tree":     "decisiontree.pkl",
    "Random Forest":     "randomforest.pkl",
    "Gradient Boosting": "gradientboosting.pkl",
}
ITEM_TYPES = [
    "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
    "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Meat",
    "Soft Drinks", "Hard Drinks", "Breads", "Breakfast", "Starchy Foods",
    "Seafood", "Others",
]
FOOD_TYPES = {
    "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy",
    "Canned", "Baking Goods", "Meat", "Breads", "Breakfast",
    "Starchy Foods", "Seafood",
}
DRINK_TYPES = {"Soft Drinks", "Hard Drinks"}
SIZE_MAP = {"Small": 0, "Medium": 1, "High": 2}
LOC_MAP  = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}

@st.cache_resource
def load_models():
    return {k: joblib.load(SAVE_DIR / v) for k, v in MODEL_FILES.items() if (SAVE_DIR / v).exists()}

@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH) if TEST_PATH.exists() else None

@st.cache_data
def load_features():
    if TRAIN_PATH.exists():
        return [c for c in pd.read_csv(TRAIN_PATH).columns if c != "Item_Outlet_Sales"]
    return []


def build_vector(weight, vis, mrp, size, loc, age, fat, itype, otype, feats):
    v = {f: 0.0 for f in feats}
    v.update(Item_Weight=weight, Item_Visibility=vis, Item_MRP=mrp, Outlet_Age=age)
    v["Price_Per_Weight"]  = mrp / weight if weight > 0 else 0
    v["Visibility_Ratio"]  = vis / 0.066
    v["Outlet_Size"]       = SIZE_MAP.get(size, 1)
    v["Outlet_Location_Type"] = LOC_MAP.get(loc, 1)
    if fat == "Regular":
        v["Item_Fat_Content_Regular"] = 1.0
    for col in (f"Item_Type_{itype}", f"Outlet_Type_{otype}"):
        if col in v:
            v[col] = 1.0
    if itype in FOOD_TYPES:
        v["Item_Category_Food"] = 1.0
    elif itype not in DRINK_TYPES:
        v["Item_Category_Non-Consumable"] = 1.0
    return np.array([[v[f] for f in feats]])

def predict_all(models, X):
    return {n: max(float(m.predict(X)[0]), 0) for n, m in models.items()}


def product_inputs(prefix):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**San pham**")
        mrp  = st.number_input("Gia (MRP $):", 30.0, 300.0, 140.0, 5.0, key=f"{prefix}_mrp")
        wt   = st.number_input("Trong luong (kg):", 4.0, 25.0, 12.0, 0.5, key=f"{prefix}_wt")
        vis  = st.slider("Visibility:", 0.01, 0.35, 0.05, 0.01, key=f"{prefix}_vis")
        fat  = st.selectbox("Fat:", ["Low Fat", "Regular"], key=f"{prefix}_fat")
        itype = st.selectbox("Loai hang:", ITEM_TYPES, key=f"{prefix}_type")
    with c2:
        st.markdown("**Cua hang**")
        otype = st.selectbox("Loai cua hang:", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"], index=1, key=f"{prefix}_out")
        size  = st.selectbox("Quy mo:", ["Small", "Medium", "High"], index=1, key=f"{prefix}_sz")
        loc   = st.selectbox("Vi tri:", ["Tier 1", "Tier 2", "Tier 3"], key=f"{prefix}_loc")
        age   = st.slider("Tuoi cua hang:", 1, 40, 20, key=f"{prefix}_age")
    return dict(weight=wt, vis=vis, mrp=mrp, size=size, loc=loc, age=age,
                fat=fat, itype=itype, otype=otype)

def render_test_tab(models, test_df):
    st.write("Chon mau tu tap test.csv de so sanh gia tri thuc te va du doan.")
    if test_df is None:
        st.warning("Khong tim thay test.csv!")
        return

    n = st.slider("So mau:", 1, min(50, len(test_df)), 10)
    start = st.slider("Tu dong:", 0, len(test_df) - n, 0)
    samples = test_df.iloc[start:start + n]
    actuals = samples["Item_Outlet_Sales"].values
    X = samples.drop(columns=["Item_Outlet_Sales"]).values

    rows = []
    for i in range(n):
        row = {"#": start + i + 1, "Thuc te": f"${actuals[i]:,.2f}"}
        for name, mdl in models.items():
            p = float(mdl.predict(X[i:i+1])[0])
            row[name] = f"${p:,.2f}"
            row[f"|Lech| {name}"] = f"${abs(p - actuals[i]):,.2f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_newproduct_tab(models, feats):
    st.write("Nhap thong so san pham moi de du doan doanh so ky vong.")
    inp = product_inputs("new")
    if not st.button("Du doan", type="primary", key="btn_new"):
        return
    X = build_vector(inp["weight"], inp["vis"], inp["mrp"], inp["size"],
                     inp["loc"], inp["age"], inp["fat"], inp["itype"], inp["otype"], feats)
    preds = predict_all(models, X)
    for name, val in preds.items():
        st.metric(name, f"${val:,.2f}")

def render_marketing_tab(models, feats):
    st.write("Mo phong tac dong cua chien dich marketing: tang trung bay, giam gia, doi kenh ban.")
    inp = product_inputs("mkt")
    st.markdown("---")
    st.markdown("**Dieu chinh Marketing**")
    boost_vis = st.slider("Visibility sau MKT:", 0.01, 0.35, min(inp["vis"] * 3, 0.30), 0.01, key="boost_vis")
    price_pct = st.slider("Dieu chinh gia (%):", -30, 20, 0, 5, key="price_pct")
    new_mrp = inp["mrp"] * (1 + price_pct / 100)
    st.caption(f"Gia sau dieu chinh: **${new_mrp:.2f}**")

    if not st.button("So sanh Truoc/Sau", type="primary", key="btn_mkt"):
        return

    X_before = build_vector(inp["weight"], inp["vis"], inp["mrp"], inp["size"],
                            inp["loc"], inp["age"], inp["fat"], inp["itype"], inp["otype"], feats)
    X_after  = build_vector(inp["weight"], boost_vis, new_mrp, inp["size"],
                            inp["loc"], inp["age"], inp["fat"], inp["itype"], inp["otype"], feats)
    before = predict_all(models, X_before)
    after  = predict_all(models, X_after)

    rows = []
    for name in models:
        b, a = before[name], after[name]
        chg = a - b
        pct = chg / b * 100 if b > 0 else 0
        rows.append({"Mo hinh": name, "Truoc": f"${b:,.2f}", "Sau": f"${a:,.2f}",
                      "Thay doi": f"${chg:+,.2f}", "%": f"{pct:+.1f}%"})
    st.table(pd.DataFrame(rows))

def main():
    st.set_page_config(page_title="BigMart Sales Prediction", page_icon="🛒", layout="wide")
    st.title("🛒 BigMart Sales Prediction Demo")
    st.write("Decision Tree · Random Forest · Gradient Boosting (tu xay dung)")

    models = load_models()
    feats  = load_features()
    test_df = load_test()
    if not models or not feats:
        st.error("Khong tim thay mo hinh hoac du lieu. Chay train_model.py truoc!")
        st.stop()

    t1, t2, t3 = st.tabs(["Test du lieu thuc te", "San pham moi", "Marketing Campaign"])
    with t1:
        render_test_tab(models, test_df)
    with t2:
        render_newproduct_tab(models, feats)
    with t3:
        render_marketing_tab(models, feats)


if __name__ == "__main__":
    main()