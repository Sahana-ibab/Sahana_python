import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def generate_data(seed=1, n=200):
    np.random.seed(seed)
    X = np.random.normal(0, 1, n)
    e =  np.random.normal(0.5, 1, n)
    return X.reshape(-1, 1), e

def generate_target(X, e):
    y = -1.1 + 0.6 * X.flatten() + e
    return y

def plot_scattered(X, y):
    plt.scatter(X, y, alpha=0.6)
    plt.title("X, y plot")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

def train_and_plot(X, y):
    X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.3, random_state=1)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # plot
    plt.scatter(X_test, y_test, label="Test Data")
    plt.plot(X_test, y_pred, linewidth=2, color="red",  label = "Regression Line")
    plt.title("test set with Regression line: ")
    plt.xlabel("X_test")
    plt.ylabel("y_test and y_pred")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model.coef_[0], model.intercept_


def main():
    X, e = generate_data()
    y = generate_target(X, e)

    print("Length of y:", len(y))  # 200
    print("True theta_0: -1.1, True theta_1: 0.6")  # Model parameters

    plot_scattered(X, y)
    theta_1, theta_0 = train_and_plot(X, y)

    print(f"Fitted theta_0 (intercept): {theta_0:.3f}")
    print(f"Fitted theta_1 (slope): {theta_1:.3f}")


if __name__ == '__main__':
    main()