# Implement Stochastic Gradient Descent algorithm from scratch

from random import randint
import pandas as pd
from sklearn.metrics import r2_score

def data_load():
    data=pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
    X=data[["age", "BMI", "BP", "blood_sugar", "Gender"]]
    y=data["disease_score_fluct"]
    return X, y

def Scaled(X,y):
    X = X.fillna(0)
    y = y.fillna(0)
    means = X.mean(axis=0)
    std_devs = X.std(axis=0)
    X_scaled = (X - means) / std_devs
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std
    return X_scaled,y_scaled

def split_data(X_scaled,y_scaled):
    train=int(0.7*(X_scaled.shape[0]))
    X_train =X_scaled.iloc[:train]
    X_test =X_scaled.iloc[train:]
    y_train=y_scaled.iloc[:train]
    y_test=y_scaled.iloc[train:]
    return X_train,X_test,y_train,y_test

def initial_theta(X_train):
    n=X_train.shape[0]
    d=X_train.shape[1]
    Theta_values=[]
    for i in range(d):
        theta = 0
        Theta_values.append(theta)
    return Theta_values

def hypothesis_func(X_train,theta_Values):
    n = X_train.shape[0]
    d = X_train.shape[1]
    y1=0
    idx = randint(0, n - 1)
    for i in range(d):
        y1+=(X_train.iloc[idx, i] * theta_Values[i])
    return y1, idx

def compute_error(y1, y_train):
    # print("err: ",y1 - y_train.iloc[idx] )
    return y1 - y_train


def Computing_gradient(error_list,X_train, idx):
    d = X_train.shape[1]
    gradient=[0] * d

    for i in range(d):
        gradient[i] = error_list * X_train.iloc[idx, i]
    return gradient

def updating_theta(gradient,theta_values):
    alpha = 0.001
    for i in range (len(theta_values)):
        theta_values[i] = theta_values[i] - alpha * gradient[i]
    return theta_values

def cost_func(error_list):
    # print("err: ", error_list)
    total_error=error_list**2
    cost_function = total_error/2
    return cost_function

def main():
    X,y = data_load()
    X_scaled , y_scaled = Scaled(X,y)
    X_train , X_test , y_train , y_test = split_data(X_scaled,y_scaled)
    theta_values = initial_theta(X_train)
    for i in range (1000):
        y1, idx = hypothesis_func(X_train,theta_values)
        # print(y1)
        error_list = compute_error(y1,y_train.iloc[idx])
        gradient = Computing_gradient(error_list,X_train, idx)
        theta_values = updating_theta(gradient, theta_values)
        cost_function=cost_func(error_list)
        print("cost:", cost_function)
    print(theta_values)

    y_pred_test = []
    for i in range(X_test.shape[0]):
        y_pred, _ = hypothesis_func(X_test, theta_values)
        y_pred_test.append(y_pred)

    error_test = [y_pred_test[i] - y_test.iloc[i] for i in range(len(y_test))]
    cost_function_test = sum(e ** 2 for e in error_test) / (2 * len(y_test))

    print(f"Final cost on the test set: {cost_function_test}")
    r2 = r2_score(y_test, y_pred_test)
    print(f"r² Score on Test Set: {r2}")


if __name__=="__main__":
    main()