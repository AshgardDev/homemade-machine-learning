import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from homemade.utils.features import prepare_for_training
from homemade.utils.hypothesis import sigmoid

class LogisticRegression_with_opt():
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
        num_features = self.data.shape[1]
        ## 分类类别
        self.unique_labels = np.unique(self.labels)
        ## 分类个数
        num_unique_labels = len(self.unique_labels)
        self.thetas = np.zeros((num_unique_labels, num_features))

    def train(self, max_iterations=500):
        cost_histories = []

        for label_index, unique_label in enumerate(self.unique_labels):
            current_labels = (self.labels == unique_label).astype(float)
            current_initial_theta = np.copy(self.thetas[label_index])

            # Run gradient descent.
            (current_theta, cost_history) = LogisticRegression_with_opt.gradient_descent(
                self.data,
                current_labels,
                current_initial_theta,
                max_iterations,
            )

            self.thetas[label_index] = current_theta.T
            cost_histories.append(cost_history)
        return self.thetas, cost_histories

    def predict(self, data):
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
        probability_predictions = LogisticRegression_with_opt.hypothesis(data_processed, self.thetas.T)
        max_probability_indices = np.argmax(probability_predictions, axis=1)
        class_predictions = np.empty(max_probability_indices.shape, dtype=object)
        num_examples = data.shape[0]
        for index, label in enumerate(self.unique_labels):
            class_predictions[max_probability_indices == index] = label

        return class_predictions.reshape((num_examples, 1))

    @staticmethod
    def gradient_descent(data, labels, initial_theta, max_iterations):
        # cost_history = []
        # new_theta = label_theta
        # for iteration in range(max_iterations):
        #     ## 计算梯度,并根据学习率,更新梯度(即更新theta, 所谓的梯度下降)
        #     new_theta = MlLogisticRegression.gradient_step(data, target, lr, new_theta)
        #     ## 计算最新的权重theta的损失,并记录
        #     cost_history.append(MlLogisticRegression.cost_function(data, target, new_theta))
        # return new_theta, cost_history

        cost_history = []
        num_features = data.shape[1]
        minification_result = minimize(
            # Function that we're going to minimize.
            lambda current_theta: LogisticRegression_with_opt.cost_function(
                data, labels, current_theta.reshape((num_features, 1))
            ),
            # Initial values of model parameter.
            initial_theta,
            # We will use conjugate gradient algorithm.
            method='CG',
            # Function that will help to calculate gradient direction on each step.
            jac=lambda current_theta: LogisticRegression_with_opt.gradient_step(
                data, labels, current_theta.reshape((num_features, 1))
            ),
            # Record gradient descent progress for debugging.
            callback=lambda current_theta: cost_history.append(LogisticRegression_with_opt.cost_function(
                data, labels, current_theta.reshape((num_features, 1))
            )),
            options={'maxiter': max_iterations}
        )
        # Reshape the final version of model parameters.
        optimized_theta = minification_result.x.reshape((num_features, 1))

        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, target, theta):
        num_examples = len(data)
        y_hat = LogisticRegression_with_opt.hypothesis(data, theta)
        error = y_hat - target
        gradients = (1/num_examples) * (data.T @ error)
        return gradients.T.flatten()

    @staticmethod
    def hypothesis(data, thetas):
        return np.clip(sigmoid(data @ thetas), 1e-8, 1 - 1e-8)

    @staticmethod
    def cost_function(data, target, theta):
        num_examples = len(data)
        y_hat = LogisticRegression_with_opt.hypothesis(data, theta)
        cross_entropy = -1/num_examples * (target.T @ np.log(y_hat) + (1 - target).T @ np.log(1 - y_hat))
        return cross_entropy[0][0]

if __name__ == '__main__':
    # import pandas as pd
    # data = pd.read_csv('../../data/iris.csv')
    #
    # # List of suppported Iris classes.
    # iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']
    #
    # # Pick the Iris parameters for consideration.
    # x_axis = 'petal_length'
    # y_axis = 'petal_width'
    #
    # # Plot the scatter for every type of Iris.
    # for iris_type in iris_types:
    #     plt.scatter(
    #         data[x_axis][data['class'] == iris_type],
    #         data[y_axis][data['class'] == iris_type],
    #         label=iris_type
    #     )
    #
    # # Plot the data.
    # plt.xlabel(x_axis + ' (cm)')
    # plt.ylabel(y_axis + ' (cm)')
    # plt.title('Iris Types')
    # plt.legend()
    # plt.show()
    #
    # # Get total number of Iris examples.
    # num_examples = data.shape[0]
    #
    # # Get features.
    # x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
    # # Get labels.
    # y_train = data['class'].values.reshape((num_examples, 1))
    #
    # # Set up linear regression parameters.
    # max_iterations = 1000  # Max number of gradient descent iterations.
    # regularization_param = 0  # Helps to fight model overfitting.
    # polynomial_degree = 0  # The degree of additional polynomial features.
    # sinusoid_degree = 0  # The degree of sinusoid parameter multipliers of additional features.
    #
    # # Init logistic regression instance.
    # logistic_regression = LogisticRegression_with_opt(x_train, y_train, polynomial_degree, sinusoid_degree)
    #
    # # Train logistic regression.
    # (thetas, costs) = logistic_regression.train(max_iterations=max_iterations)
    #
    # # Print model parameters that have been learned.
    # res = pd.DataFrame(thetas, columns=['Theta 1', 'Theta 2', 'Theta 3'], index=['SETOSA', 'VERSICOLOR', 'VIRGINICA'])
    # print(res)
    #
    # # %%
    # # Draw gradient descent progress for each label.
    # labels = logistic_regression.unique_labels
    # plt.plot(range(len(costs[0])), costs[0], label=labels[0])
    # plt.plot(range(len(costs[1])), costs[1], label=labels[1])
    # plt.plot(range(len(costs[2])), costs[2], label=labels[2])
    #
    # plt.xlabel('Gradient Steps')
    # plt.ylabel('Cost')
    # plt.legend()
    # plt.show()
    #
    # # Make training set predictions.
    # y_train_predictions = logistic_regression.predict(x_train)
    #
    # # Check what percentage of them are actually correct.
    # precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
    #
    # print('Precision: {:5.4f}%'.format(precision))
    #
    # # Get the number of training examples.
    # num_examples = x_train.shape[0]
    #
    # # Set up how many calculations we want to do along every axis.
    # samples = 150
    #
    # # Generate test ranges for x and y axis.
    # x_min = np.min(x_train[:, 0])
    # x_max = np.max(x_train[:, 0])
    #
    # y_min = np.min(x_train[:, 1])
    # y_max = np.max(x_train[:, 1])
    #
    # X = np.linspace(x_min, x_max, samples)
    # Y = np.linspace(y_min, y_max, samples)
    #
    # # z axis will contain our predictions. So let's get predictions for every pair of x and y.
    # Z_setosa = np.zeros((samples, samples))
    # Z_versicolor = np.zeros((samples, samples))
    # Z_virginica = np.zeros((samples, samples))
    #
    # for x_index, x in enumerate(X):
    #     for y_index, y in enumerate(Y):
    #         data = np.array([[x, y]])
    #         prediction = logistic_regression.predict(data)[0][0]
    #         if prediction == 'SETOSA':
    #             Z_setosa[x_index][y_index] = 1
    #         elif prediction == 'VERSICOLOR':
    #             Z_versicolor[x_index][y_index] = 1
    #         elif prediction == 'VIRGINICA':
    #             Z_virginica[x_index][y_index] = 1
    #
    # # Now, when we have x, y and z axes being setup and calculated we may print decision boundaries.
    # for iris_type in iris_types:
    #     plt.scatter(
    #         x_train[(y_train == iris_type).flatten(), 0],
    #         x_train[(y_train == iris_type).flatten(), 1],
    #         label=iris_type
    #     )
    #
    # plt.contour(X, Y, Z_setosa)
    # plt.contour(X, Y, Z_versicolor)
    # plt.contour(X, Y, Z_virginica)
    #
    # plt.xlabel(x_axis + ' (cm)')
    # plt.ylabel(y_axis + ' (cm)')
    # plt.title('Iris Types')
    # plt.legend()
    # plt.show()
    import pandas as pd
    data = pd.read_csv('../../data/microchips-tests.csv')
    X_train = np.array(data[['param_1', 'param_2']])
    y_train = np.array(data[['validity']])
    lrwm = LogisticRegression_with_opt(X_train, y_train, polynomial_degree=15, sinusoid_degree=15,
                                            normalize_data=False)
    lrwm.train()

