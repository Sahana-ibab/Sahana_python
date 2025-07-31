from ISLP import load_data
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_and_prepare_data():
    data = load_data('NCI60')
    X = data['data']
    y = data['labels']['label']
    print("Shape of data:", X.shape)
    print("Number of unique cancer types:", y.nunique())
    print("\nCancer type distribution:\n", y.value_counts())

    return X, y, data

def perform_pca(X, y,n=3):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)

    # DataFrame for PCA components
    pca_df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
    pca_df['CancerType'] = y.values

    # Summary statistics
    print("Summary statistics for PCA components:")
    print(pca_df.describe())

    plt.figure(figsize=(12, 4))
    for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x='CancerType', y=pc, data=pca_df, palette='Set2')
        plt.title(f'{pc} by Cancer Type')
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Correlation matrix of PCs
    corr = pca_df[['PC1', 'PC2', 'PC3']].corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation between PCA Components')
    plt.show()

    plt.figure(figsize=(12, 4))
    for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
        plt.subplot(1, 3, i + 1)
        sns.violinplot(x='CancerType', y=pc, data=pca_df, palette='Set2')
        plt.title(f'{pc} Distribution')
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return pca, X_pca

def plot_variance(pca):
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='blue',
             label='Variance Explained')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, color='red', marker='o', linestyle='--',
             label='Cumulative Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained')
    plt.title('Explained and Cumulative Variance by PCA Components')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pca_scatter(X_pca, y, data, pc_x=0, pc_y=1):
    loadings = PCA(n_components=3).fit(data['data']).components_.T
    scale = 1000

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, pc_x], y=X_pca[:, pc_y], hue=y, palette="Set2", s=100)
    plt.title(f'PCA: PC{pc_x + 1} vs PC{pc_y + 1}')
    plt.xlabel(f'Principal Component {pc_x + 1}')
    plt.ylabel(f'Principal Component {pc_y + 1}')

    for i in range(loadings.shape[0]):
        plt.arrow(0, 0,
                  scale * loadings[i, pc_x],
                  scale * loadings[i, pc_y],
                  color='gray', alpha=0.1, head_width=0.2, head_length=0.2)
        plt.text(loadings[i, pc_x] * scale * 1.05, loadings[i, pc_y] * scale * 1.05,
                 i, fontsize=8, color='gray')

    plt.legend(title='Cancer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def hierarchical_clustering(X, X_pca_subset, y):
    # Agglomerative clustering on original data
    model = AgglomerativeClustering(n_clusters=4, linkage='complete', metric='euclidean')
    model.fit(X)
    print("Agglomerative Clustering Labels:\n", model.labels_)

    # Dendrogram
    Z = linkage(X_pca_subset, method='complete')
    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=y.values, orientation='top', color_threshold=40)
    plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)')
    plt.xlabel('Cancer Types')
    plt.ylabel('Distance')
    plt.show()

    print("\nCluster assignments with 4 clusters:")
    print(cut_tree(Z, n_clusters=4).T)

    print("\nCluster assignments at height=40:")
    print(cut_tree(Z, height=40).T)

def main():
    X, y, data = load_and_prepare_data()
    pca, X_pca = perform_pca(X,y)
    plot_variance(pca)

    plot_pca_scatter(X_pca, y, data, pc_x=0, pc_y=1)
    plot_pca_scatter(X_pca, y, data, pc_x=0, pc_y=2)

    hierarchical_clustering(X, X_pca[:, :2], y)


if __name__ == '__main__':
    main()
