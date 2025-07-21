import numpy as np

class MyDecisionTree:

    def __init__(self):
        self.tree = None

    def fit(self, X, y, labels):
        self.tree = self._build_tree(X, y, labels, 1)
        return self.tree

    def _build_tree(self, X, y, labels, depth):
        ### 叶子节点该返回什么?
        if X.shape[0] == 0:
            return self._calc_majority_cnt(y)

        if len(np.unique(y)) == 1:
            return y[0]

        best_split_index, best_split_feature_name = self._best_split(X, y, labels)
        values = np.unique(X[:, best_split_index])
        tree = {
            best_split_feature_name:{}
        }
        for value in values:
            mask = X[:, best_split_index] == value
            X_sub = np.delete(X[mask], best_split_index, axis=1)
            labels_sub = np.delete(labels, best_split_index)
            y_sub = y[mask]

            ## 构造其他分支子树
            tree[best_split_feature_name][value] = self._build_tree(X_sub, y_sub, labels_sub, depth + 1)

        return tree

    def _best_split(self, X, y, feature_names):
        best_feature_index = self._best_feature_to_split(X, y)
        return best_feature_index, feature_names[best_feature_index]

    def _calc_type(self, y):
        types = np.unique(y)
        counts = []
        for type in types:
            mask = y == type
            counts.append(len(y[mask]))
        return np.array(counts)

    def _calc_entropy(self, y):
        counts = self._calc_type(y)
        probs = counts / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _information_gain(self, X_col, y):
        base_entropy = self._calc_entropy(y)
        values = np.unique(X_col)
        split_entropy = 0.0
        for value in values:
            mask = X_col == value
            sub_y = y[mask]
            split_entropy += len(sub_y) / len(y) * self._calc_entropy(sub_y)
        return base_entropy - split_entropy

    def _best_feature_to_split(self, X, y):
        num_features = X.shape[1]
        gains = [self._information_gain(X[:, i], y) for i in range(num_features)]
        return np.argmax(gains)

    def _calc_majority_cnt(self, y):
        counts = self._calc_type(y)
        return counts[np.argmax(counts)]


def clean_numpy_str(obj):
    if isinstance(obj, dict):
        return {clean_numpy_str(k): clean_numpy_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_str(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_numpy_str(i) for i in obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    else:
        return str(obj)


if __name__ == '__main__':
    # X = np.array([
    #     ['成年', 'man', "高", "篮球"],
    #     ['未成年', 'man', "中", "足球"],
    #     ['成年', 'man', "低", "羽毛球"],
    #     ['未成年', 'man', "低", "足球"],
    #     ['成年', 'man', "中", "篮球"],
    #     ['未成年', 'women', "高", "足球"],
    #     ['成年', 'women', "低", "篮球"],
    #     ['未成年', 'women', "高", "羽毛球"],
    #     ['成年', 'women', "中", "羽毛球"],
    #     ['未成年', 'women', "低", "足球"],
    # ])
    # labels = np.array(["是否成年", "性别", "等级", "爱好"])
    # y = np.array(['好看', '好看', '不好看', '不好看', '好看', '不好看', '好看', '不好看', '不好看', '不好看'])
    #

    X = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 0, 1, 2],
        [1, 0, 1, 2],
        [2, 0, 1, 2],
        [2, 0, 1, 1],
        [2, 1, 0, 1],
        [2, 1, 0, 2],
        [2, 0, 0, 0]
    ])

    y = np.array(['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'])
    labels = np.array(['age', 'work', 'home', 'loan'])
    tree = MyDecisionTree()
    tree.fit(X, y, labels)
    cleaned_dict = clean_numpy_str(tree.tree)

    import json
    print(cleaned_dict)
    print(json.dumps(cleaned_dict, ensure_ascii=False, indent=2))

