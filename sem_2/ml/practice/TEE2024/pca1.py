import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def generate_data(seed=1):
    np.random.seed(seed)
    class1 = np.random.normal(loc=0.0, scale=1.0, size=(40, 50))
    class2 = np.random.normal(loc=3.0, scale=1.0, size=(40, 50))
    class3 = np.random.normal(loc=-3.0, scale=1.0, size=(40, 50))

    X = np.vstack((class1, class2, class3))

    y = np.array([0]*40+[1]*40+[2]*40)
    return X, y

def pca_plot(X, y_true):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(6,8))
    for i in np.unique(y_true):
        plt.scatter(X_pca[y_true == i, 0], X_pca[y_true == i, 1], label=f"class{i}", alpha=0.7)

    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('plotting first two principle components: ')
    plt.legend()
    plt.grid(True)
    plt.show()
    return X_pca

def run_Kmeans(X, y_true, k=3):
    kmeans= KMeans(n_clusters=k, n_init=10, random_state=1)
    y_pred = kmeans.fit_predict(X)

    print("\nCross Tab: True label v/s Kmeans Clusters: ")
    print(pd.crosstab(y_true, y_pred))


def main():
    X, y_true = generate_data()
    X_pca = pca_plot(X, y_true)
    run_Kmeans(X, y_true)



if __name__ == '__main__':
    main()



