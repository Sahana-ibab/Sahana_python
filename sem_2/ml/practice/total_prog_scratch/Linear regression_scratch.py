import numpy as np
import matplotlib.pyplot as plt

# Hypothesis function (linear regression: h(X) = Xθ)
def hypothesis(X, theta):
    return np.dot(X, theta)  # No sigmoid, just linear transformation

# Cost function (Mean Squared Error)
def compute_cost(X, y, theta):
    m = len(y)  # Number of samples
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # MSE
    return cost

# Gradient Descent function
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []  # To store cost over iterations

    for i in range(num_iterations):
        predictions = hypothesis(X, theta)
        gradient = (1/m) * np.dot(X.T, (predictions - y))  # Compute gradient
        theta -= learning_rate * gradient  # Update theta
        cost = compute_cost(X, y, theta)  # Compute cost
        cost_history.append(cost)

        if i % 100 == 0:  # Print cost every 100 iterations
            print(f"Iteration {i}: Cost = {cost}")

    return theta, cost_history

# Main function
def main():
    # Generate sample dataset (simulating a linear relationship)
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
    y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

    # Add bias term (column of ones for intercept)
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Initialize parameters
    theta = np.zeros((X.shape[1], 1))  # Two parameters (bias + weight)
    learning_rate = 0.1
    num_iterations = 1000

    # Train the model
    optimal_theta, cost_history = gradient_descent(X, y, theta, learning_rate, num_iterations)

    # Plot cost vs iterations
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.show()

    print(f"Optimal parameters (theta): {optimal_theta.ravel()}")

    # Predictions
    y_pred = hypothesis(X, optimal_theta)

    # Plot the best-fit line
    plt.scatter(X[:, 1], y, color="blue", label="Actual Data")  # Original data points
    plt.plot(X[:, 1], y_pred, color="red", label="Regression Line")  # Predicted line
    plt.xlabel("Feature X")
    plt.ylabel("Target y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()

main()
