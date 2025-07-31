import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step (a) and (b): Generate X and noise e
def generate_data(seed=1, n=200):
    np.random.seed(seed)
    X = np.random.normal(0, 1, n)           # X ~ N(0,1)
    e = np.random.normal(0, 0.5, n)         # e ~ N(0, 0.25); std = sqrt(0.25) = 0.5
    return X.reshape(-1, 1), e                       # X as column vector for sklearn

# Step (c): Generate y = -1.1 + 0.6X + e
def generate_target(X, e):
    y = -1.1 + 0.6 * X.flatten() + e                # flatten X to match e shape
    return y

# Step (d): Scatter plot of X vs y
def plot_scatter(X, y):
    plt.scatter(X, y, alpha=0.6)
    plt.title("Scatter plot of X vs y")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

# Step (e): Fit linear regression and plot line on test set
def train_and_plot(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plot
    plt.scatter(X_test, y_test, label='Test Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title("Test Set with Regression Line")
    plt.xlabel("X_test")
    plt.ylabel("y_test and y_pred")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model.coef_[0], model.intercept_

def main():
    X, e = generate_data()
    y = generate_target(X, e)

    # (c) Answer:
    print("Length of y:", len(y))  # 200
    print("True theta_0: -1.1, True theta_1: 0.6")  # Model parameters

    plot_scatter(X, y)
    theta_1, theta_0 = train_and_plot(X, y)

    print(f"Fitted theta_0 (intercept): {theta_0:.3f}")
    print(f"Fitted theta_1 (slope): {theta_1:.3f}")

if __name__ == "__main__":
    main()
