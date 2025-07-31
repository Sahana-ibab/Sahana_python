import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree

# Load and preprocess the USArrests data
df = get_rdataset('USArrests').data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ===================== (a) Complete Linkage + Euclidean =====================
link_euclidean = linkage(X_scaled, method='complete', metric='euclidean')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(link_euclidean, labels=df.index.tolist(), leaf_rotation=90)
plt.title("Dendrogram - Complete Linkage + Euclidean Distance")
plt.tight_layout()
plt.show()

# ===================== (b) Cut into 3 clusters (Euclidean) =====================
clusters_euclidean = cut_tree(link_euclidean, n_clusters=3).flatten()
# Create a pandas Series with the cluster labels
cluster_series = pd.Series(clusters_euclidean + 1, index=df.index)  # +1 to make cluster labels 1-based
# Group state names into clusters based on the cluster labels
cluster_dict = cluster_series.groupby(cluster_series).apply(list).to_dict()

print("\nClusters using Euclidean Distance:")
for cluster, states in cluster_dict.items():
    state_names = df.index[states].tolist()  # Get the state names based on the indices
    print(f"Cluster {cluster}: {state_names}")

# ===================== (c) Complete Linkage + Correlation =====================
corr_dist = pairwise_distances(X_scaled, metric='correlation')
link_corr = linkage(corr_dist, method='complete')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(link_corr, labels=df.index.tolist(), leaf_rotation=90)
plt.title("Dendrogram - Complete Linkage + Correlation Distance")
plt.tight_layout()
plt.show()

# ===================== (d) Cut into 3 clusters (Correlation) =====================
clusters_corr = cut_tree(link_corr, n_clusters=3).flatten()
# Create a pandas Series with the cluster labels
correlation_clusters = pd.Series(clusters_corr + 1, index=df.index)  # +1 to make cluster labels 1-based
# Group state names into clusters based on the cluster labels
correlation_cluster_dict = correlation_clusters.groupby(correlation_clusters).apply(list).to_dict()

print("\nClusters using Correlation Distance:")
for cluster, states in correlation_cluster_dict.items():
    state_names = df.index[states].tolist()  # Get the state names based on the indices
    print(f"Cluster {cluster}: {state_names}")

# ===================== (e) Proportionality between Correlation and Euclidean =====================
# Compute squared Euclidean distances
euclidean_sq = pairwise_distances(X_scaled, metric='euclidean') ** 2

# Compute correlation distances
correlation_distance = 1 - np.corrcoef(X_scaled)

# Get upper triangle values (excluding diagonal)
triu_indices = np.triu_indices_from(euclidean_sq, k=1)
euclidean_flat = euclidean_sq[triu_indices]
correlation_flat = correlation_distance[triu_indices]

# Scatter plot to show proportionality
plt.figure(figsize=(6, 5))
plt.scatter(correlation_flat, euclidean_flat, alpha=0.7)
plt.xlabel("1 - Correlation (rij)")
plt.ylabel("Squared Euclidean Distance")
plt.title("Proportionality: 1 - rij vs. Squared Euclidean Distance")
plt.grid(True)
plt.tight_layout()
plt.show()
