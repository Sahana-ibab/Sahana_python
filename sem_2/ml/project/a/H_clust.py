import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

def data_load():
    X = pd.read_csv("/home/ibab/datasets_IA2/USArrests.csv", index_col=0)
    return X

def perform_clustering(X, method='complete', metric='euclidean'):
    if metric == 'correlation':
        dist_matrix = pdist(X, metric='correlation')
        return linkage(dist_matrix, method=method)
    else:
        return linkage(X, method=method, metric=metric)

def plot_dendrogram(linkage_matrix, labels, title):
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=labels)
    plt.title(title)
    plt.xlabel("States")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

def get_clusters(linkage_matrix, num_clusters):
    return fcluster(linkage_matrix, num_clusters, criterion='maxclust')

def display_clusters(states, cluster_labels, method_name):
    df = pd.DataFrame({'State': states, 'Cluster': cluster_labels})
    print(f"\nClusters ({method_name}):")
    print(df.sort_values(by='Cluster'))

def main():
    X = data_load()
    x = X.values

    # Euclidean Distance
    link_euc = perform_clustering(x, method='complete', metric='euclidean')
    plot_dendrogram(link_euc, X.index, "Complete Linkage + Euclidean Distance")
    clusters_euc = get_clusters(link_euc, 3)
    display_clusters(X.index, clusters_euc, "Euclidean")

    # Correlation Distance
    link_corr = perform_clustering(x, method='complete', metric='correlation')
    plot_dendrogram(link_corr, X.index, "Complete Linkage + Correlation Distance")
    clusters_corr = get_clusters(link_corr, 3)
    display_clusters(X.index, clusters_corr, "Correlation")

if __name__ == '__main__':
    main()
