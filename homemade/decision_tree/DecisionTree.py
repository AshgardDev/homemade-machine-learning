## 分类、回归模型都支持
import numpy as np

class DecisionTree:
    def __init__(self, mode='classification', max_depth=None, min_samples_split=2, max_unique_threshold=3):
        assert mode in ['classification', 'regression']
        self.mode = mode
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_unique_threshold = max_unique_threshold
        self.tree = None
        self.labels = None

    def fit(self, X, y, labels):
        X, y = np.array(X), np.array(y)
        self.labels = [str(l) for l in labels]
        is_discrete = self._auto_detect_feature_types(X)
        self.tree = self._build_tree(X, y, self.labels, is_discrete, depth=1)
        return self.tree

    def predict(self, X):
        X = np.array(X)
        return [self._predict_sample(x, self.tree) for x in X]

    def _predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree  # 叶子节点

        feature_name = str(next(iter(tree)))
        branches = tree[feature_name]

        if feature_name not in self.labels:
            return None

        feature_index = self.labels.index(feature_name)
        feature_val = x[feature_index]

        for branch_key, subtree in branches.items():
            if isinstance(branch_key, str):
                if branch_key.startswith('<='):
                    threshold = float(branch_key.split('<=')[-1])
                    if float(feature_val) <= threshold:
                        return self._predict_sample(x, subtree)
                elif branch_key.startswith('>'):
                    threshold = float(branch_key.split('>')[-1])
                    if float(feature_val) > threshold:
                        return self._predict_sample(x, subtree)
                else:
                    # 离散值判断
                    if str(feature_val) == branch_key:
                        return self._predict_sample(x, subtree)
            else:
                # 安全兜底
                if feature_val == branch_key:
                    return self._predict_sample(x, subtree)

        # 如果没有找到匹配路径，默认返回 None
        return None


    def _build_tree(self, X, y, labels, is_discrete, depth, default_label=None):
        if X.shape[0] == 0 or len(y) == 0:
            return default_label

        if X.shape[1] == 0:
            return self._calc_leaf_output(y)

        if self.mode == 'classification' and len(np.unique(y)) == 1:
            return y[0]

        if self.max_depth is not None and depth > self.max_depth:
            return self._calc_leaf_output(y)

        if X.shape[0] < self.min_samples_split:
            return self._calc_leaf_output(y)

        current_leaf = self._calc_leaf_output(y)

        best_index, best_feature, split_info = self._best_split(X, y, labels, is_discrete)
        tree = {best_feature: {}}

        if split_info['type'] == 'discrete':
            for val in np.unique(X[:, best_index]):
                mask = X[:, best_index] == val
                X_sub = np.delete(X[mask], best_index, axis=1)
                y_sub = y[mask]
                labels_sub = np.delete(labels, best_index)
                is_discrete_sub = np.delete(is_discrete, best_index)
                tree[best_feature][f"{val}"] = self._build_tree(
                    X_sub, y_sub, labels_sub, is_discrete_sub, depth + 1, default_label=current_leaf)
        else:
            threshold = split_info['threshold']
            left_mask = X[:, best_index] <= threshold
            right_mask = ~left_mask
            tree[best_feature][f"<= {threshold}"] = self._build_tree(
                X[left_mask], y[left_mask], labels, is_discrete, depth + 1, default_label=current_leaf)
            tree[best_feature][f"> {threshold}"] = self._build_tree(
                X[right_mask], y[right_mask], labels, is_discrete, depth + 1, default_label=current_leaf)

        return tree

    def _best_split(self, X, y, labels, is_discrete):
        best_index, info = self._best_feature_to_split(X, y, is_discrete)
        return best_index, labels[best_index], info

    def _best_feature_to_split(self, X, y, is_discrete):
        infos, scores = [], []
        for i in range(X.shape[1]):
            if is_discrete[i]:
                info = self._calc_split_discrete(X[:, i], y)
            else:
                info = self._calc_split_continuous(X[:, i], y)
            infos.append(info)
            scores.append(info["score"])
        best_idx = np.argmin(scores)
        return best_idx, infos[best_idx]

    def _calc_split_discrete(self, X_col, y):
        unique_vals = np.unique(X_col)
        weighted_score = 0
        for val in unique_vals:
            mask = X_col == val
            y_sub = y[mask]
            weighted_score += len(y_sub) / len(y) * self._calc_score(y_sub)
        return {"type": "discrete", "score": weighted_score}

    def _calc_split_continuous(self, X_col, y):
        sort_idx = np.argsort(X_col)
        X_sorted = X_col[sort_idx]
        y_sorted = y[sort_idx]
        unique_vals = np.unique(X_sorted)
        if len(unique_vals) == 1:
            return {"type": "continuous", "score": float('inf'), "threshold": None}
        thresholds = (X_sorted[:-1] + X_sorted[1:]) / 2
        best_score = float("inf")
        best_threshold = None
        for t in thresholds:
            left_mask = X_sorted <= t
            right_mask = ~left_mask
            y_left, y_right = y_sorted[left_mask], y_sorted[right_mask]
            score = (
                len(y_left) / len(y_sorted) * self._calc_score(y_left) +
                len(y_right) / len(y_sorted) * self._calc_score(y_right)
            )
            if score < best_score:
                best_score = score
                best_threshold = t
        return {"type": "continuous", "score": best_score, "threshold": best_threshold}

    def _calc_score(self, y):
        if self.mode == 'classification':
            values, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)
        else:
            if len(y) == 0:
                return 0
            mean = np.mean(y)
            return np.mean((y - mean) ** 2)

    def _calc_leaf_output(self, y):
        if self.mode == 'classification':
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        else:
            return float(np.mean(y))

    def _auto_detect_feature_types(self, X):
        is_discrete = []
        for i in range(X.shape[1]):
            col = X[:, i]
            if np.issubdtype(col.dtype, np.number):
                is_discrete.append(len(np.unique(col)) <= self.max_unique_threshold)
            else:
                is_discrete.append(True)
        return np.array(is_discrete)

if __name__ == '__main__':
    X = [
        [25, 1],
        [40, 2],
        [35, 1],
        [25, 2],
        [55, 1],
        [15, 1],
        [5, 2]
    ]
    y = [100.0, 300.0, 200., 300, 200, 100, 300]
    labels = ['年龄', '等级']

    reg = DecisionTree(mode='regression', max_depth=5)

    reg.fit(X, y, labels)

    print(reg.tree)
    print(reg.predict([[30, 1], [45, 2]]))

