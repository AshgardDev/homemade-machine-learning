import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork:
    def __init__(self, hidden_layer_sizes):
        if len(hidden_layer_sizes) < 1 and hidden_layer_sizes[0] < 0:
            raise ValueError("Neural Network needs at least one hidden layer size")
        self.hidden_layer_sizes = np.array(hidden_layer_sizes)
        self.weights = [] # w0, w1, w2
        self.biases = [] # b0, b1, b2
        self.X = None
        self.y = None
        self.y_onehot = None
        self.feature_num = None
        self.class_num = None

    def train(self, X, y, max_iter=1):
        self.X = X
        self.y = y
        if X.ndim != 2:
            raise ValueError("X.ndim must be 2")
        unique_values = np.unique(y)
        self.feature_num = X.shape[1]
        self.class_num = len(unique_values)
        self.weights, self.biases = NeuralNetwork.init_weights_and_biases(self.feature_num, self.hidden_layer_sizes, self.class_num)
        self.y_onehot = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

        cost_history = []
        for epoch in range(max_iter):
            y_pred, cache = self.forward(self.weights, self.biases, self.X)
            cost = self.categorical_cross_entropy(y_pred, self.y)
            cost_history.append(cost)
            self.backward(self.weights, self.biases, self.X, cache)
        return cost_history

    def categorical_cross_entropy(self, y_pred, y_true):
        ## softmax + cce
        ## 不转出独热编码
        ## 取出正确标签位置所在的概率值
        y_pred_probs = y_pred[range(len(y_pred)), y_true]
        return -np.mean(np.log(y_pred_probs + 1e-9))

    def forward(self, weights, biases, X):
        A_cache = []
        A_cache.append(X)
        A = X
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            Z = np.dot(A, weight) + bias  # (batch_size, n_l)
            if index == len(weights) - 1:  # 最后一层
                A = self.softmax(Z)
            else:
                A = self.sigmoid(Z)
            A_cache.append(A)

        return A, A_cache

    @staticmethod
    def init_weights_and_biases(feature_num, hidden_layer_sizes, class_num):
        layer_sizes = np.hstack(([feature_num], hidden_layer_sizes, [class_num]))
        weights = [NeuralNetwork.random_weight(layer_sizes[i], layer_sizes[ i +1]) for i in range(len(layer_sizes ) -1)]
        biases = [NeuralNetwork.random_bias(layer_sizes[ j +1]) for j in range(len(layer_sizes ) -1)]
        return weights, biases

    @staticmethod
    def random_weight(start, end):
        return np.random.randn(start, end) * np.sqrt(2 / start)

    @staticmethod
    def random_bias(num):
        return np.random.randn(num) * np.sqrt(2 / num)

    def sigmoid(self, x):
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        z = np.zeros_like(x, dtype=float)

        z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        z[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
        return z

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止溢出
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def backward(self, weights, biases, X, cache):
        L = len(cache) - 1
        A = cache ## [A0, A1, A2, ..., AL]
        dW = []
        db = []

        N = len(X)
        delta_L = A[L] - self.y_onehot
        dW.append(A[L-1].T @ delta_L / N)
        db.append(delta_L.sum(axis=0, keepdims=True))

        delta = delta_L
        for l in range(L-1, 0, -1):
            delta = (delta @ self.weights[l].T) * (A[l] * (1 - A[l]))
            dW.append(A[l-1].T @ delta / N)
            db.append(delta.sum(axis=0, keepdims=True))

        dW = dW[::-1]
        db = db[::-1]
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - 0.01 * dW[i]
            self.biases[i] = self.biases[i] - 0.01 * db[i]

if __name__ == '__main__':
    print("正在下载 MNIST 数据集...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    print("MNIST 数据集下载完成！")

    X, y = mnist.data, mnist.target
    print(f"\n数据集形状 (X): {X.shape}")
    print(f"标签形状 (y): {y.shape}")

    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # %%
    nn = NeuralNetwork(hidden_layer_sizes=[16, 4])
    cost_history = nn.train(X_train[:100], y_train[:100], 200)

    plt.figure()
    plt.plot(cost_history)
    plt.show()


