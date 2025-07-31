import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# (a) Generate Simulated Data
np.random.seed(42)

# Class 1: Mean shift at (2, 2, ..., 2)
class_1 = np.random.normal(loc=2, scale=1, size=(40, 50))

# Class 2: Mean shift at (6, 6, ..., 6)
class_2 = np.random.normal(loc=6, scale=1, size=(40, 50))

# Class 3: Mean shift at (10, 10, ..., 10)
class_3 = np.random.normal(loc=10, scale=1, size=(40, 50))

# Combine the data into a single dataset
data = np.vstack([class_1, class_2, class_3])

# Create true labels for the classes
labels_true = np.array([0]*40 + [1]*40 + [2]*40)

# (b) Perform PCA
# Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

# Plot the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[labels_true == 0, 0], pca_result[labels_true == 0, 1], color='red', label='Class 1')
plt.scatter(pca_result[labels_true == 1, 0], pca_result[labels_true == 1, 1], color='green', label='Class 2')
plt.scatter(pca_result[labels_true == 2, 0], pca_result[labels_true == 2, 1], color='blue', label='Class 3')

plt.title('PCA - First Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.show()

# If the classes show separation in the plot, proceed to part (c)
# Otherwise, you can adjust the simulation parameters for greater separation.

# (c) Perform K-means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# Compare clustering results with true labels
contingency_table = pd.crosstab(labels_true, kmeans_labels, rownames=['True Class'], colnames=['Cluster'])
print("\nContingency Table comparing true labels and K-means clustering labels:")
print(contingency_table)

# Optional: Plot the K-means clusters in the 2D PCA space
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[kmeans_labels == 0, 0], pca_result[kmeans_labels == 0, 1], color='red', label='Cluster 1')
plt.scatter(pca_result[kmeans_labels == 1, 0], pca_result[kmeans_labels == 1, 1], color='green', label='Cluster 2')
plt.scatter(pca_result[kmeans_labels == 2, 0], pca_result[kmeans_labels == 2, 1], color='blue', label='Cluster 3')

plt.title('K-means Clustering Results on PCA-reduced Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.show()

# Key Observations:
#
# Perfect Separation, but Mislabeled Clusters
# Each true class is entirely assigned to a single cluster, but the cluster labels do not match the true class labels.
# True Class 0 → All 40 points in Cluster 2.
# True Class 1 → All 40 points in Cluster 0.
# True Class 2 → All 40 points in **Cluster 1`.
# K-means Arbitrary Labeling
# K-means assigns cluster numbers randomly (e.g., "Cluster 0" doesn’t inherently correspond to "True Class 0").
# Here, the mapping is:
# True Class 0 = Cluster 2
# True Class 1 = Cluster 0
# True Class 2 = Cluster 1
# Clustering Accuracy
# Despite the label mismatch, K-means perfectly separated the classes.
# Accuracy is 100% if you relabel clusters to match true classes.



##NCI60 dataset is used##

#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from scipy.cluster.hierarchy import linkage, fcluster
# from sklearn.metrics import accuracy_score
#
# from ISLP import load_data
#
# def load_nci_data():
#     NCI60 = load_data('NCI60')
#     X = NCI60['data']
#     y = NCI60['labels'].values.ravel()
#     return X, y
#
# def reduce_features_pca(X, n_components=5):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     pca = PCA(n_components=n_components)
#     X_pca = pca.fit_transform(X_scaled)
#     return X_pca
#
# def reduce_features_hclust(X, n_clusters=5):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     linkage_matrix = linkage(X_scaled.T, method='ward')
#     cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
#
#     clustered_features = []
#     for cluster_num in range(1, n_clusters + 1):
#         cluster_columns = np.where(cluster_assignments == cluster_num)[0]
#         cluster_mean = X_scaled[:, cluster_columns].mean(axis=1)
#         clustered_features.append(cluster_mean)
#
#     X_hclust = np.vstack(clustered_features).T
#     return X_hclust
#
# from sklearn.model_selection import train_test_split
#
# def evaluate_model(X, y):
#     model = SVC(kernel='rbf', C=1.0, gamma='scale')
#     model.fit(X, y)
#     y_pred = model.predict(X)
#     accuracy = accuracy_score(y, y_pred)
#     return accuracy
#
#
# def main():
#     # Load data
#     X, y = load_nci_data()
#
#     # PCA-based feature reduction
#     X_pca = reduce_features_pca(X, n_components=5)
#     pca_accuracy = evaluate_model(X_pca, y)
#     print(f'PCA-based classification accuracy (SVM): {pca_accuracy:.4f}')
#
#     # Hierarchical clustering-based feature reduction
#     X_hclust = reduce_features_hclust(X, n_clusters=5)
#     hclust_accuracy = evaluate_model(X_hclust, y)
#     print(f'Hierarchical clustering-based classification accuracy (SVM): {hclust_accuracy:.4f}')
#
#     # Visualize results
#     results_df = pd.DataFrame({
#         'Method': ['PCA (5 PCs)', 'Hierarchical Clustering (5 clusters)'],
#         'Accuracy': [pca_accuracy, hclust_accuracy]
#     })
#
#     sns.barplot(data=results_df, x='Method', y='Accuracy')
#     plt.ylim(0, 1)
#     plt.title('SVM Classification Accuracy Comparison')
#     plt.ylabel('Accuracy')
#     plt.show()
#
# if __name__ == "__main__":
#     main()


