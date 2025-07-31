import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def load_data():
    df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')
    X_vector = df[["age"]]

    y_vector = df["disease_score"]
    return X_vector, y_vector


def main():
    X_vector, y_vector=load_data()
    split_index = int(0.7 * len(X_vector))
    X_train, X_test = X_vector[:split_index], X_vector[split_index:]
    y_train, y_test = y_vector[:split_index], y_vector[split_index:]

    X_scaled = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    print(X_scaled)
    y_scaled = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)

    X_t_scaled = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    y_t_scaled = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)

    theta=np.dot((np.linalg.inv(np.dot(X_scaled.T,X_scaled))),np.dot(X_scaled.T,y_scaled))
    print("Theta: ", theta)

    y_pred=np.dot(X_t_scaled, theta)
    print(y_pred)
    r2=r2_score(y_t_scaled, y_pred)
    print(r2)

    # J= (np.dot(X_scaled,theta))-y_scaled
    # J=0.5*np.dot(J.T,J)
    # print(J)
    # n_J=(np.dot((np.dot(X_scaled.T,X_scaled)),theta))-np.dot(X_scaled.T,y_scaled)
    # print(n_J)
    # print(J)

if __name__=="__main__":
    main()