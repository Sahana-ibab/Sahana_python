import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

# Set style
sns.set_style(style="whitegrid")

# Create non-linear data using make_moons
X, y = make_moons(n_samples=200,noise=0.5, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Create DataFrames for plotting
df_train = pd.DataFrame(X_train, columns=['x1', 'x2'])
df_train['y'] = y_train
df_test = pd.DataFrame(X_test, columns=['x1', 'x2'])
df_test['y'] = y_test

# Define plot_clf function
def plot_clf(model, df, grid_range, show_contours=False, show_support_vectors=False, title=''):
    x1 = grid_range
    x2 = grid_range
    xx1, xx2 = np.meshgrid(x1, x2, sparse=False)
    Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T

    decision_boundary = model.predict(Xgrid)
    decision_boundary_grid = decision_boundary.reshape(len(x2), len(x1))

    decision_function = model.decision_function(Xgrid)
    decision_function_grid = decision_function.reshape(len(x2), len(x1))

    plt.figure(figsize=(10, 10))
    if show_contours:
        plt.contourf(x1, x2, decision_function_grid)
        plt.contour(x1, x2, decision_boundary_grid)

    sns.scatterplot(x='x1', y='x2', hue='y', data=df)
    if show_support_vectors and hasattr(model, "support_vectors_"):
        sns.scatterplot(x=model.support_vectors_[:, 0],
                        y=model.support_vectors_[:, 1],
                        color='red',
                        marker='+',
                        s=500)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Train models
linear_svc = SVC(max_iter=10000,kernel='linear')
poly_svc = SVC(kernel='poly', degree=3)
rbf_svc = SVC(kernel='rbf')

# Fit models
linear_svc.fit(X_train, y_train)
poly_svc.fit(X_train, y_train)
rbf_svc.fit(X_train, y_train)

# Evaluate on train and test
models = {'Linear SVC': linear_svc, 'Polynomial SVC (deg=3)': poly_svc, 'RBF SVC': rbf_svc}

for name, model in models.items():
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}:\n  Train Accuracy: {train_acc:.2f}\n  Test Accuracy:  {test_acc:.2f}\n")

# Plot decision boundaries
grid_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)

plot_clf(linear_svc, df_train, grid_range, show_contours=True, show_support_vectors=True, title='Linear SVC Decision Boundary')
plot_clf(poly_svc, df_train, grid_range, show_contours=True, show_support_vectors=True, title='Polynomial SVC (degree=3) Decision Boundary')
plot_clf(rbf_svc, df_train, grid_range, show_contours=True, show_support_vectors=True, title='RBF SVC Decision Boundary')
