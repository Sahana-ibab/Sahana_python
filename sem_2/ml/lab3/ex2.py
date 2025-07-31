import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# compute hypothesis function:
def hypothesis_func(X_vector, theta):
    hyp_fn=np.dot(X_vector, theta)
    return hyp_fn


# error
def errors(hyp_fn, y_vector):
    e=hyp_fn-y_vector
    return e

# calculate the gradient:
def gradients_fn(X_vector,e):
    gradients=np.dot(X_vector.T,e)
    return gradients

#Update thetas:
def update_thetas(alpha, theta, gradients):
    theta=theta-(alpha*gradients)
    print(theta)
    return theta

def cost_fn(e):
    return np.sum(np.dot(e, e)) / (2 * len(e))
    # J=np.sum(np.dot(e, e))/2
    # return J

def main():
    df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')
    # print(df.describe())
    X_vector = df[["age", "BMI", "BP", "blood_sugar", "Gender"]]
    # X_vector = df[["age"]]
    y_vector = df["disease_score"]
    ones = np.ones((X_vector.shape[0], 1))
    X_vector = np.hstack((ones, X_vector))
    # df = pd.read_csv('/home/ibab/housing.csv')
    # # print(df.describe())
    # X_vector = df[["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"]]
    # y_vector = df["median_house_value"]
    # X_vector = X_vector.fillna(0)
    # y_vector = y_vector.fillna(0)
    # [X_vector, y_vector]= fetch_california_housing(return_X_y=True)
    # print(X_vector,y_vector)

    # X_vector=np.array([[1,1,2],[1,2,1],[1,3,3]])
    # y_vector=np.array([3,4,5])


    split_index = int(0.7* len(X_vector))
    X_train, X_test = X_vector[:split_index], X_vector[split_index:]
    y_train, y_test = y_vector[:split_index], y_vector[split_index:]
    theta = np.zeros(X_train.shape[1])

    # print(theta)
    alpha=0.0001
    iter=1000
    epsilon = 1e-8
    # cost function is increasing ....so ... scaling the data:
    X_scaled=(X_train-X_train.mean(axis=0))/(X_train.std(axis=0)+ epsilon)
    y_scaled = (y_train - y_train.mean(axis=0)) / (y_train.std(axis=0)+epsilon)

    X_t_scaled=(X_test-X_test.mean(axis=0))/(X_test.std(axis=0)+epsilon)
    y_t_scaled = (y_test - y_test.mean(axis=0)) / (y_test.std(axis=0)+epsilon)

    # Calling all functions:
    c=[]
    iterate=[]
    for i in range(iter):
        print("Number of iterations: ", i+1)
        hyp_fn=hypothesis_func(X_scaled, theta)
        e=errors(hyp_fn, y_scaled)
        gradients=gradients_fn(X_scaled,e)
        theta=update_thetas(alpha, theta, gradients)
        print("Updated thetas: ", theta)

        J=cost_fn(e)
        print("Cost fn: ", J, "\n")
        iterate.append(i)
        c.append(J)
    print("...",theta)
    test_hyp_fn = hypothesis_func(X_t_scaled, theta)
    print(test_hyp_fn)
    test_errors = errors(test_hyp_fn, y_t_scaled)
    test_cost = cost_fn(test_errors)
    print(f"Final cost on test set: {test_cost}")

    plt.plot(iterate, c)
    plt.show()
    print(len(y_t_scaled))
    r2=r2_score( y_t_scaled, test_hyp_fn )
    print("r2 score: ",r2)




if __name__ == '__main__':
    main()

