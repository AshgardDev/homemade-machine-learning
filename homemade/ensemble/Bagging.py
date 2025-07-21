from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import numpy as np

class MyBaggingClassifier:
    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples  # 样本比例
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.estimators_ = []
        self.random_state = random_state
        self.features_indices_ = []

    def fit(self, X, y):
        ## 固定随机状态
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        for _ in range(self.n_estimators):
            ## 样本抽样个数
            n_indices = int(self.max_samples * n_samples)
            if self.bootstrap:
                sub_indices = rng.choice(range(len(X)), n_indices, replace=True)
            else:
                sub_indices = rng.choice(range(len(X)), n_indices, replace=False)

            X_sub = X[sub_indices]
            y_sub = y[sub_indices]

            ## 特征抽样个数
            n_features = X.shape[1]
            n_features_indices = int(self.max_features * n_features)
            if n_features_indices > 0 and n_features_indices < n_features:
                sub_feature_indices = rng.choice(n_features, n_features_indices, replace=False)
                self.features_indices_.append(sub_feature_indices)
                X_sub = X_sub[:, sub_feature_indices]

            new_estimator = clone(self.base_estimator)
            new_estimator.fit(X_sub, y_sub)
            self.estimators_.append(new_estimator)

    def predict(self, X):
        y_preds = np.array([estimator.predict(X[:, self.features_indices_[pos]]) for pos, estimator in enumerate(self.estimators_)])
        row_preds = y_preds.T
        result = []
        for row in row_preds:
            type, counts = np.unique(row, return_counts=True)
            result.append(type[np.argmax(counts)])
        return np.array(result)

if __name__ == '__main__':
    bagging_clf = MyBaggingClassifier(max_samples=0.5, max_features=0.5, bootstrap=True)
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    bagging_clf.fit(X, y)
    y_pred = bagging_clf.predict(X)
    print(y_pred.reshape(-1, 50))

