import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.algorithm.decision_tree import DecisionTreeRegressorScratch

class GradientBoostingRegressorScratch:
    def __init__(self, n_estimators=15, learning_rate=0.1, max_depth=4, min_samples_split=10):
        self.n_estimators = n_estimators      # Số bước boosting (số cây)
        self.learning_rate = learning_rate    # Tốc độ học eta (bước thu hẹp sai số)
        self.max_depth = max_depth            # Cây nông (weak learner)
        self.min_samples_split = min_samples_split
        self.base_pred = 0                    # Dự đoán cơ sở ban đầu F_0
        self.trees = []                       # Danh sách các cây học phần dư

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # 1.Khởi tạo dự đoán ban đầu: F_0 = Trung bình cộng của y
        self.base_pred = np.mean(y)
        # Giá trị dự đoán hiện tại của toàn bộ mẫu
        y_curr_pred = np.full(shape=y.shape, fill_value=self.base_pred)
        self.trees = []

        for m in range(self.n_estimators):
            #2. Tính phần dư (Pseudo Residual): r = y_thực_tế - y_dự_đoán_hiện_tại
            residuals = y - y_curr_pred

            # 3.Khớp một cây quyết định con để HỌC PHẦN DƯ này
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            #Dự đoán phần dư từ cây mới
            res_pred = tree.predict(X)

            # 4.Cập nhật dự đoán hiện tại theo công thức: F_m = F_{m-1} + learning_rate * h_m(x)
            y_curr_pred += self.learning_rate * res_pred
            self.trees.append(tree)

    def predict(self, X):
        X = np.array(X, dtype=float)
        # Bắt đầu từ giá trị cơ sở F_0
        preds = np.full(shape=(X.shape[0],), fill_value=self.base_pred)
        # Cộng dồn dự đoán của từng cây nhân với learning rate
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds