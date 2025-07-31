import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
def load_data():
    data = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    # Select the features (X) and the atrget variable (y)
    X = data.iloc[:, :-2]  # except last 2 cols
    y = data.iloc[:, -2]
    return X, y #pandas
def add_bias_term(X):
    X=np.array(X) #ensure if it's a np array
    ones=np.ones((X.shape[0],1))
    X=np.hstack((ones,X))
    return X
def hypothesis(X,theta):
    #dot pdt of the features(X) and the param (theta)
    # h_theta(X)=X.theta (theta-vector of model parameters including the bias)
    return np.dot(X,theta) #numpy array
def compute_cost(X,y,theta):
    m=len(y)
    predictions=hypothesis(X,theta)
    cost=(1/(2*m))*np.sum((predictions-y)**2)
    return cost
def split_data(X,y,test_size=0.30):
    np.random.seed(42)
    num_samples=X.shape[0]
    indices=np.random.permutation(num_samples)
    split_index=int(num_samples*(1-test_size))
    X_train,X_test=X.iloc[indices[:split_index]],X.iloc[indices[split_index:]]
    y_train, y_test = y.iloc[indices[:split_index]], y.iloc[indices[split_index:]]
    return X_train,X_test,y_train,y_test #pandas df and pd series respectively
def compute_gradient(X_i,y_i,theta):
    predictions=hypothesis(X_i,theta)
    gradient=np.dot(X_i.T,(predictions-y_i))
    return gradient #gradient is a numpy array
def stochastic_gd(X,y,theta,alpha,no_of_iter):
    m=len(y)
    costs=[]
    y = np.array(y).reshape(-1, 1)  # Convert y to NumPy array and reshape
    for i in range(no_of_iter):
        for j in range(m):
            index=np.random.randint(m) #firstly choosing a random sample
            X_i=X[index].reshape(1,-1) #extract a random sample and reshaping to a row vector
            y_i=y[index].reshape(-1,1) #reshaping the corresponding col
            #gradient computation
            gradient=compute_gradient(X_i,y_i,theta)
            #update the theta
            theta=theta-alpha*gradient
            #compute the cost
        if i % 100 == 0:
            cost=compute_cost(X,y,theta)
            costs.append(cost)
            print(f"Iteration {i}: Cost{cost}")
    return theta,costs
def scale_data(X):
    for col in X.columns:
        std=X[col].std()
        mean=X[col].mean()
        if std!=0:
            X[col]=(X[col]-mean)/std
        else:
            X[col]=X[col]
    return X #pandas df
def main():
    X,y=load_data()
    #splitting the data
    X_train,X_test,y_train,y_test=split_data(X,y,test_size=0.30) #pd df
    #scaling
    X_train_scaled=scale_data(pd.DataFrame(X_train))
    X_test_scaled=scale_data(pd.DataFrame(X_test)) #pd df
    #adding bias term
    X_train=add_bias_term(X_train_scaled)
    X_test=add_bias_term(X_test_scaled) #np array
    #theta init
    theta=np.zeros((X_train.shape[1],1)) #if we give full X dim error
    #hyperparameter
    alpha=0.001
    no_of_iter=1000
    opt_theta,costs=stochastic_gd(X_train,y_train,theta,alpha,no_of_iter) #np arrays
    print(f"Optimal theta:{opt_theta}")
    #evaluation
    y_pred=hypothesis(X_test,opt_theta)
    y_test = y_test.to_numpy().reshape(-1, 1)
    #r2_score
    r2=r2_score(y_test,y_pred)
    print(f"r2:{r2}")
    #mse
    mse=mean_squared_error(y_test,y_pred)
    print(f"MSE:{mse}")
main()
