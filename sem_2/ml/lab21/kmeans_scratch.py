# Implement K-Means algorithm ground-up using Python
import numpy as np

# Initializing centroids randomly
def initialize_centroids(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]


# Computing Euclidean distance between points and centroids
def compute_distances(X, centroids):
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)


# Update centroids by calculating the mean of assigned points
def update_centroids(X, labels, K):
    centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
    return centroids


# K-means algorithm
def kmeans(X, K, max_iters=100):
    centroids = initialize_centroids(X, K)
    for i in range(max_iters):
        # Step 2: Assign clusters
        distances = compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)

        # Step 3: Update centroids
        new_centroids = update_centroids(X, labels, K)

        # Convergence check (if centroids don't change, stop)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


# Example data (2D for simplicity)
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Apply K-means
K = 2
centroids, labels = kmeans(X, K)

print("Centroids:\n", centroids)
print("Labels:", labels)













