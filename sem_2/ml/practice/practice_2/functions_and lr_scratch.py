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
    m=len(y)
    predictions=hypothesis(X,theta)
    cost=(1/(2*m))*np.sum((predictions-y)**2)
    return cost
def compute_gradient(X,y,theta):
    m=len(y)
    predictions=hypothesis(X,theta)
    gradient=(1/m)*np.dot(X.T,(predictions-y))
    return gradient
def gradient_descent(X,y,theta,learning_rate,no_of_iter):
    costs=[] #for storing costs
    for i in range(no_of_iter):
        gradient=compute_gradient(X,y,theta)
        theta=theta-learning_rate*gradient #update theta
        cost=compute_cost(X,y,theta)
        costs.append(cost)
        if i%100==0:
            print(f"Iteration {i}: Cost {cost}")
    y_train_mean=np.mean(y)
    #sst-total sum of squares
    sst=np.sum((y-y_train_mean)**2)
    #ssr residual sum of squares
    ssr=np.sum((hypothesis(X,theta)-y)**2)
    #r2_score
    r2_scores=1-(ssr/sst)
    print(f"r2:{r2_scores}")
    return theta,costs
def split_data(X,y,test_size=0.30):
    np.random.seed(42)
    num_samples=X.shape[0]
    indices=np.random.permutation(num_samples)
    split_index=int(num_samples*(1-test_size))
    X_train,X_test=X[indices[:split_index]],X[indices[split_index:]]
    y_train,y_test=y[indices[:split_index]],y[indices[split_index:]]
    return X_train,X_test,y_train,y_test
# def scale_data(X):
#     for col in X.columns:
#         std=X[col].std()
#         mean=X[col].mean()
#         if std!=0:
#             X[col]=(X[col]-mean)/std
#         else:
#             X[col]=X[col]
#     return X
#
# def main():
#     X,y=load_data()
#
#     X_train,X_test,y_train,y_test=split_data(X.values,y.values,test_size=0.30)
#     X_train=scale_data(pd.DataFrame((X_train),columns=X.columns))
#     X_test=scale_data(pd.DataFrame((X_test),columns=X.columns))
#     X_train=add_bias_term(X_train.values)
#     X_test=add_bias_term(X_test.values)
#     theta=np.zeros(X_train.shape[1])
#     learning_rate=0.01
#     no_of_iter=500
#     opt_theta,costs=gradient_descent(X_train,y_train, theta, learning_rate, no_of_iter)
#     print(f"OPtimal theta:{opt_theta}")
#     y_pred=hypothesis(X_test,opt_theta)
#     mse=np.mean((y_test-y_pred)**2)
#     print(f"MSE is {mse}")
#     plt.plot(range(len(costs)),costs)
#     plt.xlabel("No of iterations")
#     plt.ylabel("Costs")
#     plt.title("Convergence of Gradient descent")
#     plt.show()
#     #test vs pred
#     plt.scatter(y_pred,y_test,color='green',marker='*',label='Predicted vs Actual values')
#     plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red',linestyle='solid',label='perfect prediction line')
#     plt.xlabel('actual values (y_test)')
#     plt.ylabel('predicted values (y_pred)')
#     plt.title('y_test vs y_pred')
#     plt.legend()
#     plt.show()
#     return opt_theta
# main()
#or if i use standardization here
def standardized_data(X):
    X_mean = np.mean(X, axis=0)  # Mean along each column
    X_std = np.std(X, axis=0)   # Std dev along each column
    X_standardized = (X - X_mean) / X_std
    return X_standardized
def main():
    X, y = load_data()

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X.values, y.values, test_size=0.30)

    # Apply standardization
    X_train = standardized_data(X_train)
    X_test = standardized_data(X_test)

    # Add bias term
    X_train = add_bias_term(X_train)
    X_test = add_bias_term(X_test)

    # Initialize parameters
    theta = np.zeros(X_train.shape[1])
    learning_rate = 0.01
    no_of_iter = 500

    # Perform gradient descent
    opt_theta, costs = gradient_descent(X_train, y_train, theta, learning_rate, no_of_iter)
    print(f"Optimal theta: {opt_theta}")

    # Evaluate the model
    y_pred = hypothesis(X_test, opt_theta)
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"MSE is {mse}")

    # Plot the cost vs iterations
    plt.plot(range(len(costs)), costs)
    plt.xlabel("No of iterations")
    plt.ylabel("Costs")
    plt.title("Convergence of Gradient Descent")
    plt.show()

    # Plot test vs predicted values
    plt.scatter(y_pred, y_test, color='green', marker='*', label='Predicted vs Actual values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='solid', label='Perfect Prediction Line')
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('y_test vs y_pred')
    plt.legend()
    plt.show()

    return opt_theta

main()