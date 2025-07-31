import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def load_data():
    data = pd.read_csv('/home/ibab/Downloads/data.csv')
    print(data.columns)
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    return X,y

# Sigmoid function (numerically stable)
def sigmoid(z):
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))
def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

# Cost function (avoid log(0) with epsilon)
def compute_cost(X, y, theta):
    m = len(y)  # Number of samples
    predictions = hypothesis(X, theta)
    epsilon = 1e-8  # Small constant to avoid log(0)
    cost = -(1 / m) * np.sum(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    return cost
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []  # To store the cost at each iteration

    for i in range(num_iterations):
        predictions = hypothesis(X, theta)
        gradient = (1 / m) * np.dot(X.T, (predictions - y))  # Gradient computation
        theta -= learning_rate * gradient  # Update theta
        cost = compute_cost(X, y, theta)  # Compute the cost for current theta
        cost_history.append(cost)

        if i % 100 == 0:  # Print cost every 100 iterations
            print(f"Iteration {i}: Cost = {cost}")

    return theta, cost_history
def add_bias_term(X):
    ones = np.ones((X.shape[0], 1))
    print(X.shape)
    print(ones.shape)
    X = np.hstack((ones, X))
    return X
def split_data(X,y,test_size=0.30,random_state=42):
    np.random.seed(random_state)
    num_samples=X.shape[0]
    indices=np.random.permutation(num_samples)
    split_index=int(num_samples*(1-test_size))
    X_train,X_test=X.iloc[indices[:split_index]],X.iloc[indices[split_index:]]
    y_train,y_test=y.iloc[indices[:split_index]],y.iloc[indices[split_index:]]
    return X_train,X_test,y_train,y_test
# Main function
def main():
    X, y = load_data()

    # Split and scale the data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.30, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add bias term
    X_train = add_bias_term(X_train)
    X_test = add_bias_term(X_test)

    # Initialize parameters
    theta = np.zeros(X_train.shape[1])  # Initialize weights to zero
    learning_rate = 0.01  # Reduced learning rate
    num_iterations = 1000

    # Train the model using gradient descent
    optimal_theta, cost_history = gradient_descent(X_train, y_train.values, theta, learning_rate, num_iterations)

    # Plot cost vs iterations
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.show()

    print(f"Optimal parameters (theta): {optimal_theta}")

    # Predictions and evaluation
    y_test_pred = (hypothesis(X_test, optimal_theta) >= 0.5).astype(int)
    accuracy = np.mean(y_test_pred == y_test.values) * 100
    print(f"Test Accuracy: {accuracy}%")
main()