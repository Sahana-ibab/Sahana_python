import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd


def generate_data(seed=1):
    np.random.seed(seed)
    # Generate 3 classes with 40 observations and 50 features each
    class1 = np.random.normal(loc=0.0, scale=1.0, size=(40, 50))  # centered at 0
    class2 = np.random.normal(loc=3.0, scale=1.0, size=(40, 50))  # shifted to 3
    class3 = np.random.normal(loc=-3.0, scale=1.0, size=(40, 50))  # shifted to -3

    # Stack all class data
    X = np.vstack((class1, class2, class3))

    # True class labels
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    return X, y


def plot_pca(X, y_true):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for i in np.unique(y_true):
        plt.scatter(X_pca[y_true == i, 0], X_pca[y_true == i, 1], label=f"Class {i}", alpha=0.7)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA - First Two Principal Components")
    plt.legend()
    plt.grid(True)
    plt.show()
    return X_pca


def run_kmeans(X, y_true, k=3):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1)
    y_pred = kmeans.fit_predict(X)

    print("\nCrosstab: True labels vs KMeans clusters")
    print(pd.crosstab(y_true, y_pred))


def main():
    X, y_true = generate_data()
    X_pca = plot_pca(X, y_true)
    run_kmeans(X, y_true)


if __name__ == '__main__':
    main()
