import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

class GBDTRegressorWithSklearnTree:
    """
    一个使用 Scikit-learn DecisionTreeRegressor 作为弱学习器的 GBDT 回归器。
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth # 控制弱学习器（树）的复杂度
        self.initial_prediction = None # F0(x)
        self.trees = [] # 存储所有弱学习器

    def fit(self, X, y):
        n_samples = X.shape[0]

        # 1. 初始化 F0(x) 为目标变量的平均值
        self.initial_prediction = np.mean(y)
        # current_predictions 存储当前集成模型的累积预测
        current_predictions = np.full(n_samples, self.initial_prediction)

        # 2. 迭代训练弱学习器
        for i in range(self.n_estimators):
            # 计算当前模型的残差 (对于MSE损失，这正是负梯度)
            residuals = y - current_predictions

            # 训练一个新的弱学习器（决策树）来拟合这些残差
            # max_depth 参数控制了每个弱学习器的复杂度，通常设置为较小的值（如1-5）
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=i) # 确保每次迭代的随机性可控
            tree.fit(X, residuals)
            self.trees.append(tree)

            # 获取新树的预测
            tree_predictions = tree.predict(X)

            # 更新当前模型的预测： F_m(x) = F_{m-1}(x) + nu * h_m(x)
            current_predictions += self.learning_rate * tree_predictions

    def predict(self, X):
        # 初始预测
        predictions = np.full(X.shape[0], self.initial_prediction)

        # 累加所有弱学习器的预测
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# --- 示例使用 ---
if __name__ == "__main__":
    # 构造一些数据，模拟一个非线性关系
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0) # 100个样本，1个特征
    y = np.sin(X).ravel() * 10 + X.ravel() + np.random.normal(0, 0.5, X.shape[0]) # 加上一些噪音和线性趋势

    # 实例化 GBDT 回归器，使用 Scikit-learn 的树
    # 注意：max_depth通常设置为一个较小的值，使每个树成为“弱学习器”
    gbdt_sklearn_tree = GBDTRegressorWithSklearnTree(n_estimators=100, learning_rate=0.1, max_depth=3)

    # 训练模型
    gbdt_sklearn_tree.fit(X, y)

    # 进行预测
    y_pred = gbdt_sklearn_tree.predict(X)

    print("GBDT 模型（使用 Scikit-learn 决策树）训练完成。")

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label='True Values', s=20, alpha=0.7)
    plt.plot(X, y_pred, color='red', label='GBDT Predictions', linewidth=2)
    plt.title('GBDT Regressor with Scikit-learn Trees')
    plt.xlabel('Feature X')
    plt.ylabel('Target y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 也可以在新的数据点上进行预测
    X_new = np.array([[0.5], [2.5], [4.5]])
    y_new_pred = gbdt_sklearn_tree.predict(X_new)
    print("\n新数据点的预测:")
    for i in range(len(X_new)):
        print(f"X: {X_new[i, 0]:.2f}, Predicted y: {y_new_pred[i]:.2f}")

