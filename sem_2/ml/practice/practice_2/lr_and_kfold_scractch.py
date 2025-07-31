#see this at the end
# import numpy as np
# import pandas as pd
#
# # Load dataset
# def load_data():
#     data = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
#     X = data.iloc[:, :-2]
#     y = data.iloc[:, -2]
#     return X, y
#
# # Normalize data using Min-Max scaling
# def normalize_data(X):
#     X_min = X.min()
#     X_max = X.max()
#     return (X - X_min) / (X_max - X_min)
#
# # Add bias term (intercept)
# def add_bias(X):
#     return np.c_[np.ones((X.shape[0], 1)), X]  # Adding a column of ones
#
# # Linear Regression: Compute Cost (MSE)
# def compute_cost(X, y, theta):
#     m = len(y)
#     predictions = X.dot(theta)
#     cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
#     return cost
#
# # Linear Regression: Gradient Descent
# def gradient_descent(X, y, theta, learning_rate, iterations):
#     m = len(y)
#     cost_history = []
#
#     for i in range(iterations):
#         gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
#         theta -= learning_rate * gradients
#         cost_history.append(compute_cost(X, y, theta))
#
#         if i % 100 == 0:
#             print(f"Iteration {i}: Cost {cost_history[-1]:.3f}")
#
#     return theta, cost_history
#
# # Manually implement K-Fold Cross-Validation
# def k_fold_cross_validation(X, y, k=10, learning_rate=0.01, iterations=1000):
#     indices = np.arange(len(X))  # Generate indices
#     np.random.shuffle(indices)  # Shuffle dataset
#     fold_size = len(X) // k  # Compute fold size
#
#     mse_scores = []
#
#     for i in range(k):
#         # Create train-test splits manually
#         start = i * fold_size
#         end = start + fold_size if i != k-1 else len(X)  # Last fold gets remaining data
#         test_idx = indices[start:end]
#         train_idx = np.setdiff1d(indices, test_idx)
#
#         # Split data
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#
#         # Train Linear Regression Model from scratch
#         theta = np.zeros(X_train.shape[1])
#         opt_theta, _ = gradient_descent(X_train, y_train, theta, learning_rate, iterations)
#
#         # Predict and evaluate using MSE
#         y_pred = X_test.dot(opt_theta)
#         mse = np.mean((y_test - y_pred) ** 2)
#         mse_scores.append(mse)
#         print(f"Fold {i+1} MSE: {mse:.3f}")
#
#     # Compute average MSE and standard deviation
#     mean_mse = np.mean(mse_scores)
#     std_dev = np.std(mse_scores)
#     print(f"\nAverage MSE: {mean_mse:.3f}")
#     print(f"Standard Deviation: {std_dev:.3f}")
#
# # Main function
# def main():
#     X, y = load_data()
#     X = normalize_data(X)
#     X = add_bias(X.values)  # Convert to NumPy array and add bias
#     k_fold_cross_validation(X, y, k=10, learning_rate=0.01, iterations=1000)
#
# main()
