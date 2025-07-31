import numpy as np
import matplotlib.pyplot as plt

# Hypothesis function (linear regression: h(X) = Xθ)
def hypothesis(X, theta):
    return np.dot(X, theta)  # No sigmoid, just linear transformation

# Cost function (Mean Squared Error) - Optional for tracking loss
def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # MSE
    return cost

# Stochastic Gradient Descent function
def stochastic_gradient_descent(X, y, theta, learning_rate, num_epochs):
    m = len(y)
    cost_history = []  # Store cost over epochs

    for epoch in range(num_epochs):
        for i in range(m):
            rand_index = np.random.randint(m)  # Pick a random sample
            x_i = X[rand_index:rand_index+1]  # Extract one sample (row)
            y_i = y[rand_index:rand_index+1]  # Corresponding label

            prediction = hypothesis(x_i, theta)
            gradient = x_i.T * (prediction - y_i)  # Compute gradient for one sample
            theta -= learning_rate * gradient  # Update theta

        # Store cost after each epoch
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if epoch % 10 == 0:  # Print cost every 10 epochs
            print(f"Epoch {epoch}: Cost = {cost}")

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
    learning_rate = 0.01  # Smaller learning rate for SGD
    num_epochs = 100  # Number of passes over dataset

    # Train the model using Stochastic Gradient Descent
    optimal_theta, cost_history = stochastic_gradient_descent(X, y, theta, learning_rate, num_epochs)

    # Plot cost vs epochs
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence (SGD)")
    plt.show()

    print(f"Optimal parameters (theta): {optimal_theta.ravel()}")

    # Predictions
    y_pred = hypothesis(X, optimal_theta)

    # Plot the best-fit line
    plt.scatter(X[:, 1], y, color="blue", label="Actual Data")  # Original data points
    plt.plot(X[:, 1], y_pred, color="red", label="Regression Line")  # Predicted line
    plt.xlabel("Feature X")
    plt.ylabel("Target y")
    plt.title("Linear Regression Fit using SGD")
    plt.legend()
    plt.show()

main()
