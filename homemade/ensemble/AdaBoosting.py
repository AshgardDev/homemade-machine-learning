from sklearn.tree import DecisionTreeClassifier
import numpy as np

class CustomAdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = [] ## 评估器权重

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        ## 重设预测值 确保y值「-1, 1」AdaBoost的sign要求
        y = np.where(y == 0, -1, 1)
        ## 初始化权重值
        sample_weights = 1/n_samples * np.ones(n_samples)
        for _ in range(self.n_estimators):
            ## 决策树桩--最简单的弱学习器,不怕弱,因为多,且越来越强
            stump = DecisionTreeClassifier(max_depth=1, random_state=np.random.randint(0, 10000))
            stump.fit(X, y, sample_weight=sample_weights)
            y_hat = stump.predict(X)
            err_mask= y != y_hat
            ## 计算错误率
            err = np.sum(sample_weights[err_mask]) / np.sum(sample_weights)
            ## 计算评估器(决策树桩)的权重
            if 0 < err < 1:
                alpha = self.learning_rate * 0.5 * np.log((1 - err) / (err + 1e-10))
            else:
                ## alpha=0,代表跳过该学习器,权重不会更新
                alpha = 0
            ## 更新样本的权重
            sample_weights = sample_weights * np.exp(-alpha * y * y_hat)
            sample_weights /= np.sum(sample_weights) + 1e-10

            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)

    def predict(self, X):
        y_preds = np.array([estimator.predict(X) for estimator in self.estimators_])
        return np.where(np.sum(y_preds.T * np.array(self.estimator_weights_), axis=1) >= 0, 1, 0)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_breast_cancer

    # Generate sample data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train custom random forest classifier
    rf = CustomAdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"CustomAda Accuracy: {accuracy:.4f}")

    from sklearn.ensemble import AdaBoostClassifier
    abc = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    abc.fit(X_train, y_train)
    y_pred = abc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Sklearn Ada Accuracy: {accuracy:.4f}")