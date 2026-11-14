import numpy as np
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature        # Vị trí cột đặc trưng dùng để chia nhánh
        self.threshold = threshold    # Ngưỡng giá trị dùng để rẽ nhánh (x <= threshold)
        self.left = left              # Nhánh con bên trái
        self.right = right            # Nhánh con bên phải
        self.value = value            # Giá trị dự đoán (nếu nút này là Nút Lá)
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=6, min_samples_split=10):
        self.max_depth = max_depth                  # Độ sâu tối đa của cây (tránh overfitting)
        self.min_samples_split = min_samples_split  # Số mẫu tối thiểu để tiếp tục chia nhánh
        self.root = None                            # Nút gốc ban đầu

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        # Bắt đầu xây dựng cây từ nút gốc đệ quy xuống dưới
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        # Điều kiện dừng đệ quy để tạo NÚT LÁ:
        # 1.Đạt độ sâu tối đa
        # 2. Số lượng mẫu còn lại quá ít
        # 3. Tất cả giá trị y đều giống hệt nhau (phương sai = 0)
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) == 0:
            leaf_value = np.mean(y)  # Dự đoán = Trung bình cộng các giá trị y trong lá
            return Node(value=leaf_value)

        # Tìm đặc trưng và ngưỡng chia tốt nhất làm giảm MSE nhiều nhất
        best_feat, best_thresh = self._best_split(X, y, n_features)

        # Nếu không tìm được cách chia nào tốt hơn thì dừng lại làm nút lá
        if best_feat is None:
            return Node(value=np.mean(y))

        # Chia dữ liệu thành 2 nhánh: Trái (<= ngưỡng) và Phải (> ngưỡng)
        left_idxs = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
        right_idxs = np.argwhere(X[:, best_feat] > best_thresh).flatten()

        # Đệ quy tiếp tục xây dựng cây con bên trái và bên phải
        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_feat, split_thresh = None, None
        parent_var = np.var(y)  # Phương sai (MSE) của nút cha hiện tại

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            # Lấy các mốc phân vị (quantiles) để tính toán nhanh thay vì thử tất cả dòng
            thresholds = np.percentile(X_column, np.linspace(10, 90, 15))

            for thresh in thresholds:
                # Tính độ giảm phương sai (Variance Reduction)
                gain = self._variance_reduction(y, X_column, thresh, parent_var)
                if gain > best_gain:
                    best_gain = gain
                    split_feat = feat_idx
                    split_thresh = thresh
        return split_feat, split_thresh

    def _variance_reduction(self, y, X_column, thresh, parent_var):
        left_mask = X_column <= thresh
        right_mask = ~left_mask
        # Nếu 1 nhánh không có mẫu nào thì cách chia này vô nghĩa
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        n = len(y)
        n_l, n_r = np.sum(left_mask), np.sum(right_mask)
        var_l, var_r = np.var(y[left_mask]), np.var(y[right_mask])
        # Trọng số phương sai của 2 nhánh con
        child_var = (n_l / n) * var_l + (n_r / n) * var_r
        return parent_var - child_var

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)