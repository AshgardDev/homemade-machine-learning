import numpy as np

class ThreeNeuralNetwork:
    def __init__(self, input_size, layer1_size, layer2_size, output_size):
        self.input_size = input_size
        self.layer1_size = layer1_size ## 5
        self.layer2_size = layer2_size ## 4
        self.output_size = output_size
        self.lr = 0.01

        ## 3 x 5
        # self.w1 = np.random.randn(self.input_size, self.layer1_size)
        ## 5 x 4
        # self.w2 = np.random.randn(self.layer1_size, self.layer2_size)
        ## 4 x 2
        # self.w3 = np.random.randn(self.layer2_size, self.output_size)

        self.w1 = np.random.randn(self.input_size, self.layer1_size) * np.sqrt(1 / self.input_size)
        self.w2 = np.random.randn(self.layer1_size, self.layer2_size) * np.sqrt(1 / self.layer1_size)
        self.w3 = np.random.randn(self.layer2_size, self.output_size) * np.sqrt(1 / self.layer2_size)

        self.b1 = np.random.randn(1, self.layer1_size)
        self.b2 = np.random.randn(1, self.layer2_size)
        self.b3 = np.random.randn(1, self.output_size)

    def sigmoid(self, x):
        # 数值稳定版 sigmoid
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        z = np.zeros_like(x, dtype=float)

        z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        z[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
        return z

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, x, y, max_iter=100):
        if x.shape[1] != self.input_size:
            raise ValueError("Input dimension does not match with input size")

        cost_history = []
        for _ in range(max_iter):
            y_pred, cache = self.forword(x)
            cost = self.binary_cross_entropy(y, y_pred)
            self.backword(x, cache, y)  # 这里调用 backword
            cost_history.append(cost)

        return cost_history

    def binary_cross_entropy(self, y_true, y_pred):
        ## 二元交叉熵的损失函数
        m = y_true.shape[0]
        # 避免log(0)
        eps = 1e-9
        loss = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        return loss

    def predict(self, x, threshold=0.5):
        y_pred_prob, _ = self.forword(x)
        return (y_pred_prob > threshold).astype(int)

    def forword(self, x):
        ## x (m, 3)
        ## a1 (m, 5)
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.sigmoid(z1)

        ## a2 (m, 4)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.sigmoid(z2)

        ## a3 (m, 2)
        z3 = np.dot(a2, self.w3) + self.b3
        a3 = self.sigmoid(z3)

        cache = {
            'a1': a1,
            'a2': a2,
            'a3': a3,
            'z1': z1,
            'z2': z2,
            'z3': z3,
        }

        return a3, cache

    def backword(self, x, cache, y):
        m = x.shape[0]
        a1, a2, a3 = cache['a1'], cache['a2'], cache['a3']

        # 错误: 使用了二元预测值y_pred
        # dz3 = y_pred - y

        # 正确: 使用sigmoid的原始输出a3
        dz3 = a3 - y

        dw3 = np.dot(a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = np.dot(dz3, self.w3.T) * self.sigmoid_derivative(a2)
        dw2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.w2.T) * self.sigmoid_derivative(a1)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 更新参数
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

# 修改你的主函数
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from matplotlib import pyplot as plt

    X, y = load_breast_cancer(return_X_y=True)
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 添加数据标准化步骤
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    network = ThreeNeuralNetwork(X_train_scaled.shape[1], 64, 32, 1)
    # 使用标准化后的数据进行训练
    cost_histories = network.fit(X_train_scaled, y_train, max_iter=2000)

    # 画损失曲线
    plt.plot(cost_histories)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.show()

    # 使用标准化后的数据进行预测
    y_pred = network.predict(X_test_scaled)
    acc = (y_pred == y_test).mean()
    print(f"Test Accuracy: {acc:.4f}")

