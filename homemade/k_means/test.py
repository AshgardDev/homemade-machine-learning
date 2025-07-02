import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Ml_KMeans():
    def __init__(self, data, n_clusters: int):
        self.data = data
        self.n_clusters = n_clusters

    def train(self, max_iter: int = 100):
        ## 初始化中心点
        centroids = Ml_KMeans.centroids_init(data=self.data, n_clusters=self.n_clusters)
        for _ in range(max_iter):
            ## 计算样本距离(中心点列表中)最近的中心点索引
            closet_centroids_ids = Ml_KMeans.centroids_find_closet(data=self.data, centroids=centroids)
            ## 更新中心点位置
            centroids = Ml_KMeans.find_centroids(self.data, centroids, closet_centroids_ids)
        return centroids, closet_centroids_ids

    @staticmethod
    def centroids_init(data, n_clusters):
        ## 随机中心点, 中心点的个数=预设集群数量, 列数等于样本列数
        random_ids = np.random.permutation(data.shape[0])[:n_clusters]
        return data[random_ids, :]

    @staticmethod
    def centroids_find_closet(data, centroids):
        num_samples = data.shape[0]
        n_clusters = centroids.shape[0]
        ## 对每个样本进行遍历,标记它所属的最近的中心点的index
        centroid_distances = np.zeros((num_samples, n_clusters))
        for data_index in range(num_samples):
            for centroid_index in range(n_clusters):
                distance_diff = centroids[centroid_index] - data[data_index]
                centroid_distances[data_index, centroid_index] = np.sum(distance_diff ** 2)
        closet_centroids_ids = centroid_distances.argmin(axis=1)
        return closet_centroids_ids

    @staticmethod
    def find_centroids(data, centroids, closet_centroids_ids):
        n_clusters = centroids.shape[0]
        centroids = np.zeros((n_clusters, data.shape[1]))
        for i in range(n_clusters):
            cluster_ids = np.where(closet_centroids_ids == i)[0]
            centroids[i] = np.average(data[cluster_ids], axis=0)
        return centroids


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    # print(iris_data.target_names)
    # print(iris_data.feature_names)

    X, y = load_iris(return_X_y=True)
    kmeans = Ml_KMeans(X, n_clusters=3)
    centroids, closet_centroids_ids = kmeans.train(max_iter=100)

    X_train = iris_data.data[:, [2, 3]]
    y_train = iris_data.target

    print(closet_centroids_ids.shape)
    print(y_train.shape)
    print(closet_centroids_ids)
    print(y_train)

    plt.figure(figsize=(12, 8), dpi=30)
    plt.subplot(121)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training data')
    plt.subplot(122)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=closet_centroids_ids, cmap='viridis', label='cluster data')
    plt.show()

