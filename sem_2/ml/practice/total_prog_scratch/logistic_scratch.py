import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hypothesis function
def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

# Cost function
def compute_cost(X, y, theta):
    m = len(y)  # Number of samples
    predictions = hypothesis(X, theta)
    # Cross entropy loss
    cost = -( 1 /m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Gradient descent function
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []  # To store the cost at each iteration

    for i in range(num_iterations):
        predictions = hypothesis(X, theta)
        gradient = ( 1 /m) * np.dot(X.T, (predictions - y))  # Gradient computation
        theta -= learning_rate * gradient  # Update theta
        cost = compute_cost(X, y, theta)  # Compute the cost for current theta
        cost_history.append(cost)

        if i % 100 == 0:  # Print cost every 100 iterations
            print(f"Iteration {i}: Cost = {cost}")

    return theta, cost_history

# Main function to train the logistic regression model
def main():
    # Example dataset
    np.random.seed(42)
    X = np.random.rand(100, 2)  # 100 samples, 2 features
    y = (np.sum(X, axis=1) > 1).astype(int)  # Label: 1 if sum of features > 1, else 0

    # Adding bias term to X
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add column of ones for bias term

    # Initialize parameters
    theta = np.zeros(X.shape[1])  # Initialize weights to zero
    learning_rate = 0.1
    num_iterations = 1000

    # Train the model using gradient descent
    optimal_theta, cost_history = gradient_descent(X, y, theta, learning_rate, num_iterations)

    # Plot cost vs iterations
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.show()

    print(f"Optimal parameters (theta): {optimal_theta}")

    # Predictions and evaluation
    predictions = (hypothesis(X, optimal_theta) >= 0.5).astype(int)
    accuracy = np.mean(predictions == y) * 100
    print(f"Training Accuracy: {accuracy}%")

main()
