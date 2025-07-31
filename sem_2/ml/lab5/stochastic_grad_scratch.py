from random import randint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def data_load():
    data = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
    X = data[["age", "BMI", "BP", "blood_sugar", "Gender"]]
    y = data["disease_score"]
    return X, y


def Scaled(X, y):
    X = X.fillna(0)
    y = y.fillna(0)
    means = X.mean(axis=0)
    std_devs = X.std(axis=0)
    X_scaled = (X - means) / std_devs
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std
    return X_scaled, y_scaled


def split_data(X_scaled, y_scaled):
    train = int(0.7 * (X_scaled.shape[0]))
    X_train = X_scaled.iloc[:train]
    X_test = X_scaled.iloc[train:]
    y_train = y_scaled.iloc[:train]
    y_test = y_scaled.iloc[train:]

    # Add bias term (column of ones) to X
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    return X_train, X_test, y_train, y_test


def initial_theta(X_train):
    d = X_train.shape[1]  # Including bias term
    return np.zeros(d)  # Initialize theta as a numpy array of zeros


def hypothesis_func(X, theta_values):
    return np.dot(X, theta_values)  # Vectorized implementation


def compute_error(y_train, y_pred):
    return y_pred - y_train  # Compute element-wise error


def compute_gradient(error, X_sample):
    return error * X_sample  # Compute gradient for a single sample


def updating_theta(gradient, theta_values, alpha=0.01):  # Increased alpha
    return theta_values - alpha * gradient  # Update rule


def cost_func(error_list):
    total_error = np.sum(error_list ** 2)
    return total_error / (2 * len(error_list))  # Average squared error


def main():
    X, y = data_load()
    X_scaled, y_scaled = Scaled(X, y)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
    theta_values = initial_theta(X_train)

    epochs = 1000
    for _ in range(epochs):
        idx = randint(0, len(y_train) - 1)  # Random sample for SGD
        X_sample = X_train[idx]  # Extract one sample
        y_sample = y_train.iloc[idx]

        y_pred = hypothesis_func(X_sample, theta_values)
        error = compute_error(y_sample, y_pred)
        gradient = compute_gradient(error, X_sample)
        theta_values = updating_theta(gradient, theta_values)

        # Print cost occasionally
        if _ % 100 == 0:
            cost = cost_func(error)
            print(f"Iteration {_}: Cost = {cost}")

    # Test set predictions
    y_pred_test = hypothesis_func(X_test, theta_values)
    error_test = compute_error(y_test, y_pred_test)
    cost_test = cost_func(error_test)
    print(f"Final Cost on Test Set: {cost_test}")

    # Plot true vs predicted values
    plt.scatter(y_test, y_pred_test)
    plt.xlabel("True Values (Scaled)")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.show()

    # Compute R² score
    r2 = r2_score(y_test, y_pred_test)
    print(f"R² Score: {r2}")


if __name__ == '__main__':
    main()
