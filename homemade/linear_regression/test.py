import numpy as np
from scipy.ndimage import histogram
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from homemade.utils.features import prepare_for_training

class MLLinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
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

        num_features = self.data.shape[1]
        ## 初始化权重
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations=500):
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations=500):
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        num_examples = self.data.shape[0]
        pred = MLLinearRegression.hypothesis(self.data, self.theta)
        ## 残差
        delta = pred - self.labels
        theta = self.theta
        theta = theta - alpha * (1/num_examples) * (self.data.T @ delta)
        self.theta = theta

    def cost_function(self, data, labels):
        num_examples = data.shape[0]
        delta = MLLinearRegression.hypothesis(data, self.theta) - labels
        cost = (1 / (2 * num_examples)) * (delta.T @ delta)
        return cost[0][0]

    def get_cost(self, data, labels):
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
        return MLLinearRegression.hypothesis(data_processed, self.theta)

    @staticmethod
    def hypothesis(data, theta):
        pred = data @ theta
        return pred


if __name__ == '__main__':
    data = pd.read_csv('../../data/world-happiness-report-2017.csv')
    data.head(10)

    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    input_param_name = 'Economy..GDP.per.Capita.'
    input_param_name2 = 'Freedom'
    output_param_name = 'Happiness.Score'

    x_train = train_data[[input_param_name, input_param_name2]].values
    y_train = train_data[[output_param_name]].values

    x_test = test_data[[input_param_name, input_param_name2]].values
    y_test = test_data[[output_param_name]].values

    # plt.figure(figsize=(10, 10), dpi=60)
    # plt.scatter(x_train, y_train, label='Training dataset')
    # plt.scatter(x_test, y_test, label='Test dataset')
    # plt.xlabel(input_param_name)
    # plt.ylabel(output_param_name)
    # plt.title('Countries Happiness')
    # plt.legend()
    # plt.show()

    learning_rate = 0.01
    num_iterations = 500
    linear_regression = MLLinearRegression(x_train, y_train)

    (theta, cost_history) = linear_regression.train(learning_rate, num_iterations)
    print('开始损失', cost_history[0])
    print('优化后的损失', cost_history[-1])

