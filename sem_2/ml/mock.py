# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
#
# import warnings
# warnings.filterwarnings("ignore")
#
# # ---------- Load and Preprocess ----------
# def load_and_preprocess(path='/home/ibab/datasets/heart.csv'):
#     df = pd.read_csv(path)
#     print("\n Dataset Loaded")
#     print(df.info())
#     print(df.describe())
#     print("\nMissing values:\n", df.isnull().sum())
#
#     X = df.drop('output', axis=1)
#     y = df['output']
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     return df, X_scaled, y, scaler
#
# # ---------- Visualize Data ----------
# def perform_eda(df):
#     sns.countplot(x='output', data=df)
#     plt.title('Target Class Distribution')
#     plt.show()
#
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
#     plt.title("Correlation Heatmap")
#     plt.show()
#
# # ---------- Cross-validation wrapper ----------
# def evaluate_model(model, X, y, name='Model'):
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#     scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
#     print(f"\n📊 {name} Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
#     return model.fit(X, y)
#
# # ---------- Confusion Matrix ----------
# def plot_confusion_matrix(model, X_test, y_test, title='Confusion Matrix', cmap='Blues'):
#     y_pred = model.predict(X_test)
#     ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=cmap)
#     plt.title(title)
#     plt.show()
#
# # ---------- Feature Importance ----------
# def plot_feature_importance(model, feature_names, title='Feature Importance'):
#     if hasattr(model, 'feature_importances_'):
#         importances = pd.Series(model.feature_importances_, index=feature_names)
#         importances.sort_values(ascending=True).plot(kind='barh')
#         plt.title(title)
#         plt.tight_layout()
#         plt.show()
#
# # ---------- Main ----------
# def main():
#     df, X_scaled, y, scaler = load_and_preprocess()
#     perform_eda(df)
#
#     # Split for confusion matrix visualization
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
#
#     # Logistic Regression
#     log_model = LogisticRegression(max_iter=1000)
#     log_model = evaluate_model(log_model, X_scaled, y, "Logistic Regression")
#     plot_confusion_matrix(log_model, X_test, y_test, "Logistic Regression")
#
#     # Random Forest
#     rf_params = {
#         'n_estimators': [100, 150, 200],
#         'max_depth': [None, 5, 10],
#         'min_samples_split': [2, 5, 10]
#     }
#     rf = RandomForestClassifier(random_state=42)
#     rf_search = RandomizedSearchCV(rf, rf_params, cv=3, n_iter=5, random_state=42, scoring='accuracy')
#     rf_search.fit(X_scaled, y)
#     best_rf = rf_search.best_estimator_
#     best_rf = evaluate_model(best_rf, X_scaled, y, "Random Forest")
#     plot_confusion_matrix(best_rf, X_test, y_test, "Random Forest", cmap='Greens')
#     plot_feature_importance(best_rf, df.columns[:-1], "Random Forest Feature Importance")
#
#     # XGBoost
#     xgb_params = {
#         'n_estimators': [100, 150],
#         'max_depth': [3, 4, 5],
#         'learning_rate': [0.01, 0.1, 0.2]
#     }
#     xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
#     xgb_search = RandomizedSearchCV(xgb, xgb_params, cv=3, n_iter=5, random_state=42, scoring='accuracy')
#     xgb_search.fit(X_scaled, y)
#     best_xgb = xgb_search.best_estimator_
#     best_xgb = evaluate_model(best_xgb, X_scaled, y, "XGBoost")
#     plot_confusion_matrix(best_xgb, X_test, y_test, "XGBoost", cmap='Oranges')
#     plot_feature_importance(best_xgb, df.columns[:-1], "XGBoost Feature Importance")
#
# # Run everything
# main()
#

# # -----------------------------------------------------------------------
# # Question 1: Logistic Regression with ROC & PR (No sklearn)
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from statsmodels.datasets import get_rdataset
#
#
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
# def train_logistic_regression(X, y, lr=0.01, epochs=1000):
#     X = np.c_[np.ones(X.shape[0]), X]  # add bias term
#     weights = np.zeros(X.shape[1])
#     for _ in range(epochs):
#         z = np.dot(X, weights)
#         h = sigmoid(z)
#         gradient = np.dot(X.T, (h - y)) / y.size
#         weights -= lr * gradient
#     return weights
#
# def predict_proba(X, weights):
#     X = np.c_[np.ones(X.shape[0]), X]
#     return sigmoid(np.dot(X, weights))
#
# def plot_roc_pr(y_true, y_scores):
#     from sklearn.metrics import roc_curve, precision_recall_curve, auc
#
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(fpr, tpr, label='ROC curve')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(recall, precision, label='PR curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.legend()
#     plt.show()
#
# # Usage
# def run_logistic_model():
#     df = pd.read_csv("/home/ibab/datasets/heart.csv")
#     X = df.drop('output', axis=1).values
#     y = df['output'].values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     weights = train_logistic_regression(X_train, y_train)
#     y_scores = predict_proba(X_test, weights)
#     plot_roc_pr(y_test, y_scores)
#
# # Question 2: Simulated 2-Class Non-Linear Dataset
# from sklearn.datasets import make_moons
# import matplotlib.pyplot as plt
#
# def generate_simulated_data():
#     X, y = make_moons(n_samples=200, noise=0.2)
#     plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
#     plt.title("Simulated Two-Class Dataset")
#     plt.show()
#     return X, y
#
# #  Question 3: SVM + SVC with Kernels
#
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
#
# def run_svm_kernels(X, y):
#     kernels = ['linear', 'rbf', 'poly', 'sigmoid']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#     for kernel in kernels:
#         model = SVC(kernel=kernel)
#         model.fit(X_train, y_train)
#         score = model.score(X_test, y_test)
#         print(f"Kernel: {kernel}, Accuracy: {score:.2f}")
#
# #  Question 4: KMeans from Scratch (Simple)
# def kmeans(X, k, iterations=100):
#     np.random.seed(42)
#     centroids = X[np.random.choice(len(X), k, replace=False)]
#     for _ in range(iterations):
#         distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
#         labels = np.argmin(distances, axis=1)
#         new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
#         if np.all(centroids == new_centroids):
#             break
#         centroids = new_centroids
#     return labels, centroids
#
# # Question 5: USArrests – Hierarchical Clustering
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from sklearn.datasets import fetch_openml
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns
#
#
# def hierarchical_clustering_usarrests(linkage_type='complete', metric='euclidean'):
#     df = get_rdataset('USArrests').data
#     df.dropna(inplace=True)
#     data = StandardScaler().fit_transform(df)
#
#     Z = linkage(data, method=linkage_type, metric=metric)
#     dendrogram(Z, labels=df.index.tolist(), leaf_rotation=90)
#     plt.title(f'Dendrogram ({linkage_type} linkage, {metric} distance)')
#     plt.show()
#
#     clusters = fcluster(Z, 3, criterion='maxclust')
#     cluster_df = pd.DataFrame({'State': df.index, 'Cluster': clusters})
#     print(cluster_df.sort_values('Cluster'))
#
#
# # Example:
# # hierarchical_clustering_usarrests('complete', 'correlation')
#
#
#
# def main():
#     # Question 1: Logistic Regression (manual ROC & PR)
#     print("\nRunning Logistic Regression with ROC and PR curves...")
#     run_logistic_model()
#
#     # Question 2: Simulated Dataset
#     print("\nGenerating Simulated Dataset...")
#     X_sim, y_sim = generate_simulated_data()
#
#     # Question 3: SVM with Different Kernels
#     print("\nRunning SVM with kernel exploration on simulated data...")
#     run_svm_kernels(X_sim, y_sim)
#
#     # Question 4: KMeans from scratch
#     print("\nRunning KMeans Clustering on simulated data...")
#     labels, centroids = kmeans(X_sim, 2)
#     print("Cluster centers:\n", centroids)
#
#     # Question 5: Hierarchical Clustering on USArrests
#     print("\nRunning Hierarchical Clustering on USArrests dataset...")
#     hierarchical_clustering_usarrests('complete', 'euclidean')
#     # You can also try: hierarchical_clustering_usarrests('complete', 'correlation')
#
# # Don't forget to call main()
# if __name__ == "__main__":
#     main()





#  Generate simulated data:
#
# We’ll create 60 samples, 20 for each class, each with 50 features.
# We'll use
# Gaussian distribution with a mean shift to make them separable.

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# np.random.seed(42)
#
# # Create 3 classes with 20 samples each, 50 features
# class_1 = np.random.normal(loc=0.0, scale=1.0, size=(20, 50))
# class_2 = np.random.normal(loc=3.0, scale=1.0, size=(20, 50))
# class_3 = np.random.normal(loc=6.0, scale=1.0, size=(20, 50))
#
# X = np.vstack((class_1, class_2, class_3))
# y = np.array([0]*20 + [1]*20 + [2]*20)  # true labels
#
# # Perform PCA and plot first 2 principal components:
#
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
#
# # Plot
# plt.figure(figsize=(8, 6))
# for label in np.unique(y):
#     plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=f'Class {label}')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA: First 2 Components')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
#
# # K-Means Clustering and comparison with true labels:
#
# from sklearn.cluster import KMeans
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
#
# kmeans = KMeans(n_clusters=3, random_state=42)
# cluster_labels = kmeans.fit_predict(X_scaled)
#
# # Compare with true labels using crosstab
# df = pd.DataFrame({'True': y, 'Cluster': cluster_labels})
# ct = pd.crosstab(df['True'], df['Cluster'])
# print(ct)
#
# # Optional: Heatmap for easier visualization
# sns.heatmap(ct, annot=True, cmap="Blues", fmt="d")
# plt.title("True Labels vs. K-Means Cluster Assignments")
# plt.ylabel("True Class")
# plt.xlabel("Cluster Label")
# plt.show()


# # Q1: Simulate a two-class dataset with visible but non-linear separation
#
# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
#
# # Step 1: Generate moon-shaped data (non-linear separation)
# X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#
# # Step 2: Convert to DataFrame for plotting
# df_train = pd.DataFrame(X_train, columns=["x1", "x2"])
# df_train["y"] = y_train
#
#
# # Function to plot decision boundary
# def plot_clf(model, X, y, title=""):
#     x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
#     x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
#     xx1, xx2 = np.meshgrid(x1_range, x2_range)
#     Xgrid = np.c_[xx1.ravel(), xx2.ravel()]
#
#     preds = model.predict(Xgrid).reshape(xx1.shape)
#
#     plt.figure(figsize=(8, 6))
#     plt.contourf(xx1, xx2, preds, alpha=0.3, cmap='coolwarm')
#     sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm')
#     plt.title(title)
#     plt.xlabel("x1")
#     plt.ylabel("x2")
#     plt.show()
#
#
# # Step 3: Train 3 SVM models - linear, polynomial, and RBF
#
# # Linear SVM
# svm_linear = SVC(kernel="linear")
# svm_linear.fit(X_train, y_train)
#
# # Polynomial SVM (degree > 1)
# svm_poly = SVC(kernel="poly", degree=3)
# svm_poly.fit(X_train, y_train)
#
# # RBF SVM
# svm_rbf = SVC(kernel="rbf")
# svm_rbf.fit(X_train, y_train)
#
# # Step 4: Accuracy scores
# print("Linear SVM - Train:", accuracy_score(y_train, svm_linear.predict(X_train)),
#       "| Test:", accuracy_score(y_test, svm_linear.predict(X_test)))
#
# print("Poly SVM - Train:", accuracy_score(y_train, svm_poly.predict(X_train)),
#       "| Test:", accuracy_score(y_test, svm_poly.predict(X_test)))
#
# print("RBF SVM - Train:", accuracy_score(y_train, svm_rbf.predict(X_train)),
#       "| Test:", accuracy_score(y_test, svm_rbf.predict(X_test)))
#
# # Step 5: Plot decision boundaries
# plot_clf(svm_linear, X_train, y_train, "Linear SVM")
# plot_clf(svm_poly, X_train, y_train, "Polynomial SVM")
# plot_clf(svm_rbf, X_train, y_train, "RBF SVM")





# Q2: PCA on NCI60 dataset from ISLP
from ISLP import load_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load NCI60 data
NCI60 = load_data('NCI60')
X = NCI60['data']
labels = NCI60['labels']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# (a) Plot PC1 vs PC2 and PC1 vs PC3 by cancer type (label color)
df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
df['CancerType'] = labels

sns.scatterplot(data=df, x='PC1', y='PC2', hue='CancerType')
plt.title("PC1 vs PC2")
plt.show()

sns.scatterplot(data=df, x='PC1', y='PC3', hue='CancerType')
plt.title("PC1 vs PC3")
plt.show()

# (b) Plot variance explained
plt.plot(pca.explained_variance_ratio_ * 100, marker='o')
plt.title("Percent Variance Explained")
plt.xlabel("Principal Component")
plt.ylabel("Variance (%)")
plt.grid(True)
plt.show()

# Cumulative variance
cumulative = pca.explained_variance_ratio_.cumsum() * 100
plt.plot(cumulative, marker='o', color='orange')
plt.title("Cumulative Variance Explained")
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Variance (%)")
plt.grid(True)
plt.show()

# (c) Hierarchical clustering on top principal components
linked = linkage(X_pca[:, :5], method='complete')  # Use first 5 PCs
plt.figure(figsize=(10, 5))
dendrogram(linked, labels=labels.values)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cell Line")
plt.ylabel("Distance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# # Q3: Manual K-Means clustering for 6 observations
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Step (a) Data
# data = np.array([[1, 4],
#                  [1, 3],
#                  [0, 4],
#                  [5, 1],
#                  [6, 2],
#                  [4, 0]])
#
# # Step (a) Plot original observations
# plt.scatter(data[:, 0], data[:, 1], c='gray')
# plt.title("Original Observations")
# for i, point in enumerate(data):
#     plt.text(point[0]+0.1, point[1], str(i+1))
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.grid(True)
# plt.show()
#
# # Step (b) Random cluster assignments
# np.random.seed(42)
# labels = np.random.choice([0, 1], size=6)
# print("Initial Cluster Labels:", labels)
#
# def compute_centroids(data, labels, k=2):
#     return np.array([data[labels == i].mean(axis=0) for i in range(k)])
#
# def assign_clusters(data, centroids):
#     distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
#     return np.argmin(distances, axis=1)
#
# # Step (c to e): Iterate until convergence
# for step in range(10):
#     old_labels = labels.copy()
#     centroids = compute_centroids(data, labels)
#     labels = assign_clusters(data, centroids)
#     if np.array_equal(labels, old_labels):
#         print(f"Converged at iteration {step+1}")
#         break
#
# # Step (f) Final plot
# colors = ['red', 'blue']
# plt.figure()
# for i in range(2):
#     cluster_points = data[labels == i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f"Cluster {i}")
# plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label="Centroids")
# plt.title("Final Clusters after K-Means")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.legend()
# plt.grid(True)
# plt.show()
