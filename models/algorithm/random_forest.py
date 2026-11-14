import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.algorithm.decision_tree import DecisionTreeRegressorScratch

class RandomForestRegressorScratch:
    def __init__(self, n_estimators=15, max_depth=6, min_samples_split=10, max_features_ratio=0.8, random_state=42):
        self.n_estimators = n_estimators              # Số lượng cây trong rừng
        self.max_depth = max_depth                    # Độ sâu tối đa mỗi cây
        self.min_samples_split = min_samples_split    # Số mẫu tối thiểu để chia
        self.max_features_ratio = max_features_ratio  # Tỷ lệ số đặc trưng chọn ngẫu nhiên cho mỗi cây
        self.random_state = random_state
        self.trees = []                               # Danh sách chứa các cây con
        self.feat_subsets = []                        # Danh sách lưu các cột đặc trưng mỗi cây sử dụng

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        self.trees = []
        self.feat_subsets = []
        # Số lượng đặc trưng lấy ngẫu nhiên cho mỗi cây
        n_sub_feats = max(1, int(n_features * self.max_features_ratio))

        for i in range(self.n_estimators):
            # 1.Kỹ thuật Bootstrap: Chọn ngẫu nhiên n_samples có lặp lại (with replacement)
            boot_idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[boot_idxs]
            y_boot = y[boot_idxs]

            # 2.Kỹ thuật Feature Subsampling: Chọn ngẫu nhiên một tập con các cột
            feat_idxs = np.random.choice(n_features, size=n_sub_feats, replace=False)
            X_boot_sub = X_boot[:, feat_idxs]

            # 3.Huấn luyện cây quyết định con
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot_sub, y_boot)

            # Lưu cây và tập đặc trưng của cây đó
            self.trees.append(tree)
            self.feat_subsets.append(feat_idxs)

    def predict(self, X):
        X = np.array(X, dtype=float)
        tree_preds = []
        for tree, feat_idxs in zip(self.trees, self.feat_subsets):
            # Chỉ lấy các cột tương ứng mà cây đó đã học
            pred = tree.predict(X[:, feat_idxs])
            tree_preds.append(pred)
        #Lấy trung bình cộng (Ensemble Average)
        return np.mean(tree_preds, axis=0)