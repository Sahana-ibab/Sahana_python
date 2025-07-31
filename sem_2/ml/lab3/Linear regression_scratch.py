import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hypothesis_func(X_vector, theta):
    hyp_fn = []
    for i in range(len(X_vector)):
        h = 0
        for j in range(len(theta)):
            h += X_vector[i][j] * theta[j]
        hyp_fn.append(h)
    print(hyp_fn)
    return hyp_fn


def errors(hyp_fn, y_vector):
    e = []
    for i in range(len(hyp_fn)):
        e.append(hyp_fn[i] - y_vector[i])
    print("e: " ,e)
    return e

def gradients_fn(X_vector, e):
    gradients = [0] * len(X_vector[0])
    for j in range(len(X_vector[0])):
        for i in range(len(X_vector)):
            gradients[j] += X_vector[i][j] * e[i]
    return gradients

def update_thetas(alpha, theta, gradients):
    for i in range(len(theta)):
        theta[i] -= alpha * gradients[i]
    return theta

def cost_fn(e):
    J = 0
    for i in range(len(e)):
        J += e[i] ** 2
    return J / 2

def main():
    df = pd.read_csv('/home/ibab/housing.csv')
    X_vector = df[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households","median_income"]].values.tolist()
    y_vector = df["median_house_value"].values.tolist()
    # print(df.describe())


    split_index = int(0.7 * len(X_vector))
    X_train, X_test = X_vector[:split_index], X_vector[split_index:]
    y_train, y_test = y_vector[:split_index], y_vector[split_index:]
    theta = [0] * len(X_train[0])

    alpha = 0.0001
    iter = 1000

    # Scaling the data:
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train_scaled = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    y_train_scaled = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)

    X_test_scaled = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    y_test_scaled = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)

    X_train_scaled = X_train_scaled.tolist()
    y_train_scaled = y_train_scaled.tolist()
    X_test_scaled = X_test_scaled.tolist()
    y_test_scaled = y_test_scaled.tolist()

    # Iterating to optimize the parameters:
    costs = []
    iterations = []
    for i in range(iter):
        print(f"Iteration: {i + 1}")
        hyp_fn = hypothesis_func(X_train_scaled, theta)
        e = errors(hyp_fn, y_train_scaled)
        gradients = gradients_fn(X_train_scaled, e)
        theta = update_thetas(alpha, theta, gradients)

        J = cost_fn(e)
        print(f"Updated thetas: {theta}")
        print(f"Cost: {J}\n")

        iterations.append(i)
        costs.append(J)

    # Testing the model:
    test_hyp_fn = hypothesis_func(X_test_scaled, theta)
    test_errors = errors(test_hyp_fn, y_test_scaled)
    test_cost = cost_fn(test_errors)
    print(f"Final cost on test set: {test_cost}")

    # Plotting the cost function:
    plt.plot(iterations, costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function vs. Iterations")
    plt.show()

if __name__ == '__main__':
    main()
