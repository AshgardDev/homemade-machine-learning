import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from homemade.utils.features import prepare_for_training
from homemade.utils.hypothesis import sigmoid

class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)
        self.data = data_processed
        self.labels = labels
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        self.features_mean = features_mean
        self.features_deviation = features_deviation

        ## 特征列数
        self.num_features = self.data.shape[1]
        ## 样本个数
        self.num_samples = self.data.shape[0]

        ## 分类类别
        self.unique_labels = np.unique(self.labels)
        ## 分类个数
        self.num_unique_labels = len(self.unique_labels)

        ## 150 * 6  3类

        ## 初始化权重
        ## theta怎么定义
        ## 目标: 计算每个样本的属于各个类别的概率值
        ## 样本的维度(num_features, 特征列数)
        ## 样本:
        ## [
        ##  [x00, x01, x02 ..., x0n]
        ##  [x10, x11, x12 ..., x1n]
        ##  [x20, x21, x22 ..., x2n]
        ##  ...
        ##  [xm0, xm1, xm2 ..., xmn]
        ## ]
        ## 计算一个样本属于类别1的概率p1, 属于类别2的概率p2, 属于类别3的概率p3
        ## p0 = x00*t00+x01*t01+...+x0n*t0n
        ## p1 = x00*t10+x01*t11+...+x0n*t1n
        ## p2 = x00*t20+x01*t21+...+x0n*t2n
        ## 需要计算3个类别的概率, 所以, theta的行数 = 类别数
        ## 需要有n列个参数t,才能和样本特征列数匹配上,所以,theta的列数位样本特征数
        ## 所以theta的维度就是(num_features, 类别数)
        self.theta = np.zeros((self.num_unique_labels, self.num_features))

    def train(self, lr = 0.01, max_iterations=500):
        cost_histories = []
        for label_index, label in enumerate(self.unique_labels):
            target = (self.labels == label).astype(float)
            label_theta = self.theta[label_index]
            new_theta, cost_history = LogisticRegression.gradient_descent(self.data, target, label_theta, lr, max_iterations)
            self.theta[label_index] = new_theta
            cost_histories.append(cost_history)
        return self.theta, cost_histories

    def predict(self, data):
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
        y_prob = sigmoid(data_processed @ self.theta.T)
        label_index = np.argmax(y_prob, axis=1)
        return self.unique_labels[label_index]

    def predict_target(self, data):
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree,
                                                    self.normalize_data)
        y_prob = sigmoid(data_processed @ self.theta.T)
        label_index = np.argmax(y_prob, axis=1)
        return label_index

    def get_cost(self, data, target):
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
        return LogisticRegression.cost_function(data_processed, target, self.theta)

    @staticmethod
    def gradient_descent(data, target, label_theta, lr, max_iterations):
        cost_history = []
        new_theta = label_theta
        for iteration in range(max_iterations):
            ## 计算梯度,并根据学习率,更新梯度(即更新theta, 所谓的梯度下降)
            new_theta = LogisticRegression.gradient_step(data, target, lr, new_theta)
            ## 计算最新的权重theta的损失,并记录
            cost_history.append(LogisticRegression.cost_function(data, target, new_theta))
        return new_theta, cost_history

    @staticmethod
    def gradient_step(data, target, lr, theta):
        n = len(data)
        y_hat = sigmoid(data @ theta)
        error = y_hat - target
        theta = theta - lr * (1/n) * (data.T @ error)
        return theta

    @staticmethod
    def cost_function(data, target, theta):
        n = len(data)
        y_hat = sigmoid(data @ theta)
        cross_entropy = -1/n * (target.T @ np.log(y_hat) + (1 - target).T @ np.log(1 - y_hat))
        return cross_entropy


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from matplotlib import pyplot as plt
    iris_data = load_iris(as_frame=True)
    print(iris_data.feature_names)
    data = iris_data.data[[iris_data.feature_names[2], iris_data.feature_names[3]]]
    labels = np.array([iris_data.target_names[target] for target in iris_data.target])

    data = np.array(data)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test, target_train, target_test = train_test_split(data, labels, iris_data.target, test_size=0.2, random_state=42)
    logist_reg = LogisticRegression(data=X_train, labels=y_train)
    theta, cost_histories = logist_reg.train(lr=0.1, max_iterations=1000)
    for i in range(len(X_test)):
        print(logist_reg.predict(np.array([X_test[i]])), y_test[i])

    # index = 0
    # for history in cost_histories:
    #     plt.plot(history, label='cost type={}'.format(logist_reg.unique_labels[index]))
    #     index += 1
    # plt.show()
    #
    # import seaborn as sns
    # sns.pairplot(data=iris_data.data)
    # plt.show()
    # #
    # # y_color = np.zeros(y_train.shape[0])
    # # for i in range(len(y_train)):
    # #     if y_train[i] == iris_data.target_names[0]:
    # #         y_color[i] = 0
    # #     elif y_train[i] == iris_data.target_names[1]:
    # #         y_color[i] = 1
    # #     elif y_train[i] == iris_data.target_names[2]:
    # #         y_color[i] = 2
    #
    # plt.scatter(X_train[:, 2], X_train[:, 3], c=target_train, cmap='viridis')
    # plt.show()

    min_petal_width = np.min(X_train[:, 0], axis=0)
    max_petal_width = np.max(X_train[:, 0], axis=0)
    min_petal_length = np.min(X_train[:, 1], axis=0)
    max_petal_length = np.max(X_train[:, 1], axis=0)
    print(min_petal_width, max_petal_width, min_petal_length, max_petal_length)
    pw = np.linspace(min_petal_width, max_petal_width, num=100)
    pl = np.linspace(min_petal_length, max_petal_length, num=100)
    xx, yy = np.meshgrid(np.array(pw), np.array(pl))
    grid = np.c_[xx.ravel(), yy.ravel()]
    print(grid.shape)
    z = logist_reg.predict_target(grid)
    print(z)
    z = z.reshape(xx.shape)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=target_train, cmap='viridis')
    plt.contour(xx, yy, z)
    plt.axis((0, 7, 0, 3))
    plt.show()


