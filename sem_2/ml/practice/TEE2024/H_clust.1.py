import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def data_load():
    X = pd.read_csv("/home/ibab/datasets_IA2/USArrests.csv", index_col=0 )
    return X

def perform_clustering(X, method='complete', metric='euclidean'):
    if metric == 'correlation':
        dist_matix = pdist(X, metric='correlation')
        return linkage(dist_matix, method= method)
    else:
        return linkage(X, method= method, metric= metric)


def plot_dendrogram(linkage_matrix, labels, title):
    plt.figure(figsize= (10, 6))
    dendrogram(linkage_matrix, labels= labels)
    plt.title(title)
    plt.xlabel("States")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

def get_clusters(linkage_matrix, num_clusters):
    return fcluster(linkage_matrix, num_clusters, criterion = "maxclust")

def display_cluster(States, Cluster_labels ,method_name):
    df = pd.DataFrame({'States': States, 'Cluster': Cluster_labels})
    print("\nClustering Method: ", method_name)
    print(df.sort_values(by = 'Cluster'))

def main():
    X = data_load()
    x = X.values

    # Euclidean:
    link_euc = perform_clustering(x, method='complete', metric='euclidean' )
    plot_dendrogram(link_euc, X.index, "Complete Linkage + Euclidean ")
    cluster_euc = get_clusters(link_euc, 3)
    display_cluster(X.index, cluster_euc, "Euclidean")

    # Correlation:
    link_corr = perform_clustering(x, method = 'complete', metric ='correlation')
    plot_dendrogram(link_corr, X.index, "complete Linkage + correlation")
    cluster_corr = get_clusters(link_corr, 3)
    display_cluster(X.index, cluster_corr, "Correlation")

if __name__ == '__main__':
    main()