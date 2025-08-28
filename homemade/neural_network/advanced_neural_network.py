import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

class Utils:
    # ========== 激活函数 ==========
    @staticmethod
    def sigmoid(z):
        # 防止溢出：clip z 到 [-500, 500]
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(a):
        return (a > 0).astype(float)

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)  # 防止exp溢出
        exp_scores = np.exp(z)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

class Layer:
    def __init__(self, input_size, output_size, activation="relu"):
        self.activation = activation
        # Kaiming 初始化 (He initialization)
        if activation == "relu":
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros(output_size)

        # 缓存结果
        self.Z = None
        self.A = None
        self.A_prev = None

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b
        if self.activation == "sigmoid":
            self.A = Utils.sigmoid(self.Z)
        elif self.activation == "relu":
            self.A = Utils.relu(self.Z)
        elif self.activation == "softmax":
            self.A = Utils.softmax(self.Z)
        return self.A

    def backward(self, dA, is_output=False, y = None):
        m = len(self.A_prev)
        if self.activation == "softmax" and is_output:
            dZ = self.A - y
        else:
            if self.activation == "sigmoid":
                dZ = dA * Utils.sigmoid_derivative(self.A)
            elif self.activation == "relu":
                dZ = dA * Utils.relu_derivative(self.A)
            else:
                raise ValueError("Unsupported activation")
        dW = (self.A_prev.T @ dZ) / m
        db = np.sum(dZ, axis=0) / m
        dA_prev = dZ @ self.W.T
        return dA_prev, dW, db

    def update_gradient(self, lr, dW, db):
        self.W = self.W - lr * dW
        self.b = self.b - lr * db


class AdvancedNeuralNetwork:
    def __init__(self, layer_dims, activations, lr=0.01):
        assert len(layer_dims) - 1 == len(activations)
        self.layers = []
        self.layers = []
        for i in range(len(activations)):
            self.layers.append(Layer(layer_dims[i], layer_dims[i+1], activations[i]))
        self.lr = lr

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y):
        dA = None
        grads = []
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                dA, dW, db = layer.backward(dA, is_output=True, y=y)
            else:
                dA, dW, db = layer.backward(dA)
            grads.append((dW, db))
        return grads[::-1]

    def cost_loss(self, y_hat, y):
        m = len(y_hat)
        loss = -np.sum(y * np.log(y_hat + 1e-8)) / m
        return loss

    def fit(self, X, y, epochs=1000, batch_size=64, verbose=100):
        m = len(X)
        cost_history = []
        for epoch in range(1, epochs+1):
            ## 打乱洗牌
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation, :]
            y_shuffled = y[permutation, :]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size, :]
                y_batch = y_shuffled[i:i+batch_size, :]

                self.forward(X_batch)
                grads = self.backward(y_batch)
                self.update_params(grads)

            ## 损失计算,不用每次迭代都跑一次,可以根据需要计算当前损失
            if epoch % verbose == 0:
                y_hat_full = self.forward(X)
                loss = self.cost_loss(y_hat_full, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                cost_history.append(loss)

        return cost_history

    def update_params(self, grads):
        for layer, (dW, db) in zip(self.layers, grads):
            layer.update_gradient(self.lr, dW, db)


if __name__ == '__main__':
    print("正在下载 MNIST 数据集...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    print("MNIST 数据集下载完成！")

    X, y = mnist.data, mnist.target
    print(f"\n数据集形状 (X): {X.shape}")
    print(f"标签形状 (y): {y.shape}")

    y = y.astype(np.uint8)
    y_one_hot = np.zeros((len(y), 10))
    y_one_hot[np.arange(len(y)), y] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ann = AdvancedNeuralNetwork([784, 32, 16, 10], ["relu", "relu", "softmax"], lr=0.01)

    print("\n--- 正在开始训练 ---")
    cost_history = ann.fit(X_train_scaled, y_train, epochs=200, batch_size=100, verbose=10)

    plt.figure()
    plt.plot(cost_history)
    plt.xlabel('Epochs (x10)')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

    print("\n--- 正在评估模型性能 ---")
    predictions = ann.predict(X_test_scaled)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test_labels) * 100
    print(f"模型在测试集上的准确率: {accuracy:.2f}%")