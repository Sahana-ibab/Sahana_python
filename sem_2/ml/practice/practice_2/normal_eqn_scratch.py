import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def load_data():
    data=pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    X=data.iloc[:,:-2]
    y=data.iloc[:,-2]
    return X,y
def add_bias_term(X):
    ones=np.ones((X.shape[0],1))
    print(X.shape)
    print(ones.shape)
    X=np.hstack((ones,X))
    return X
def hypothesis(X,theta):
    return np.dot(X,theta)
def compute_cost(X,y,theta):
    hyp=hypothesis(X,theta)
    cost=np.dot((hyp-y).T,(hyp-y))
    return cost
def normal_eqn(X,y):
    first=np.dot(X.T,X)
    inverse=np.linalg.inv(first)
    second=np.dot(X.T,y)
    theta=np.dot(inverse,second)
    return theta
def split_data(X,y,test_size=0.30):
    np.random.seed(42)
    num_samples=X.shape[0]
    indices=np.random.permutation(num_samples)
    split_index=int(num_samples*(1-test_size))
    X_train,X_test=X[indices[:split_index]],X[indices[split_index:]]
    y_train,y_test=y[indices[:split_index]],y[indices[split_index:]]
    return X_train,X_test,y_train,y_test
def scale_data(X):
    for col in X.columns:
        std=X[col].std()
        mean=X[col].mean()
        if std!=0:
            X[col]=(X[col]-mean)/std
        else:
            X[col]=X[col]
    return X
def main():
    X,y=load_data()
    X_train,X_test,y_train,y_test=split_data(X.values,y.values,test_size=0.30)
    #scaling
    X_train_scaled=scale_data(pd.DataFrame(X_train))
    X_test_scaled=scale_data(pd.DataFrame(X_test))
    #adding bias term
    X_train=add_bias_term(X_train_scaled)
    X_test=add_bias_term(X_test_scaled)
    #theta init
    theta=np.zeros(X_train.shape[1])
    #hyperparameter
    alpha=0.001
    no_of_iter=1000
    opt_theta=normal_eqn(X_train,y_train)
    print(f"OPtimal theta:{opt_theta}")
    y_pred=hypothesis(X_test,opt_theta)
    #r2score
    y_test_mean=np.mean(y_test)
    sst=np.sum((y_test-y_test_mean)**2)
    ssr=np.sum((y_pred-y_test)**2)
    r2_scores=1-(ssr/sst)
    print(f"r2 score is {r2_scores}")
    #mse
    MSE=np.mean((y_test-y_pred)**2)
    print(f"MSE:{MSE}")
main()