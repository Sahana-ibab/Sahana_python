# K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.
# import pandas as pd
# import numpy as np
# from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
# # compute hypothesis function:
# def hypothesis_func(X_vector, theta):
#     hyp_fn=np.dot(X_vector, theta)
#     return hyp_fn
#
# # error
# def errors(hyp_fn, y_vector):
#     e=hyp_fn-y_vector
#     # print("e: ",e)
#     return e
#
# # calculate the gradient:
# def gradients_fn(X_vector,e):
#     gradients=np.dot(X_vector.T,e)
#     return gradients
#
# #Update thetas:
# def update_thetas(alpha, theta, gradients):
#     theta=theta-(alpha*gradients)
#     # print(theta)
#     return theta
#
# def cost_fn(e):
#     return np.sum(np.dot(e, e)) / (2 * len(e))
#     # J=np.sum(np.dot(e, e))/2
#     # return J
#
# def main():
#
#     df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')
#     shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#     # print(df.describe())
#     X_vector = shuffled_df[["age", "BMI", "BP", "blood_sugar", "Gender"]]
#     # X_vector = df[["age"]]
#     y_vector = shuffled_df["disease_score"]
#     ones = np.ones((X_vector.shape[0], 1))
#     X_vector = np.hstack((ones, X_vector))
#     f=(len(X_vector)//10)
#     epsilon = 1e-8
#
#     # cost function is increasing ....so ... scaling the data:
#     X = (X_vector - X_vector.mean(axis=0)) / (X_vector.std(axis=0) + epsilon)
#     y = (y_vector - y_vector.mean(axis=0)) / (y_vector.std(axis=0) + epsilon)
#     cost_dict={}
#     for k in range(0,10):
#         X_train, X_test = np.vstack((X[:k*f],X[(k+1)*f:])), X[k*f:(k+1)*f]
#         y_train, y_test = np.hstack((y[:k*f],y[(k+1)*f:])), y[k*f:(k+1)*f]
#         print(np.mean(X_train))
#
#         theta = np.zeros(X_train.shape[1])
#
#         # print(theta)
#         alpha=0.0001
#         iter=1000
#
#         # Calling all functions:
#         c=[]
#         iterate=[]
#         for i in range(iter):
#             # print("Number of iterations: ", i+1)
#             hyp_fn=hypothesis_func(X_train, theta)
#             e=errors(hyp_fn, y_train)
#             gradients=gradients_fn(X_train,e)
#             theta=update_thetas(alpha, theta, gradients)
#             # print("Updated thetas: ", theta)
#
#             J=cost_fn(e)
#             # print("Cost fn: ", J, "\n")
#             iterate.append(i)
#             c.append(J)
#         print(f"fold: {k+1}")
#         print("Thetas: ",theta)
#         test_hyp_fn = hypothesis_func(X_test, theta)
#         # print(test_hyp_fn)
#         test_errors = errors(test_hyp_fn, y_test)
#         test_cost = cost_fn(test_errors)
#         print(f"Final cost on test set: {test_cost}")
#         cost_dict[k+1]=test_cost
#         plt.plot(iterate, c)
#         # plt.show()
#         print(len(y_test))
#
#         r2=r2_score( y_test, test_hyp_fn )
#         print("r2 score: ",r2,"\n")
#     print("Cost values of different folds: ",cost_dict)
#
# if __name__ == '__main__':
#     main()


# K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# compute hypothesis function:
def hypothesis_func(X_vector, theta):
    hyp_fn=np.dot(X_vector, theta)
    return hyp_fn

# error
def errors(hyp_fn, y_vector):
    e=hyp_fn-y_vector
    # print("e: ",e)
    return e

# calculate the gradient:
def gradients_fn(X_vector,e):
    gradients=np.dot(X_vector.T,e)
    return gradients

#Update thetas:
def update_thetas(alpha, theta, gradients):
    theta=theta-(alpha*gradients)
    # print(theta)
    return theta

def cost_fn(e):
    return np.sum(np.dot(e, e)) / (2 * len(e))
    # J=np.sum(np.dot(e, e))/2
    # return J

def main():

    df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print(df.describe())
    X_vector = shuffled_df[["age", "BMI", "BP", "blood_sugar", "Gender"]]
    # X_vector = df[["age"]]
    y_vector = shuffled_df["disease_score"]
    ones = np.ones((X_vector.shape[0], 1))
    X_vector = np.hstack((ones, X_vector))
    f=(len(X_vector)//10)
    epsilon = 1e-8

    # cost function is increasing ....so ... scaling the data:

    cost_dict={}
    for k in range(0,10):
        print(f"fold: {k + 1}")
        X_train1, X_test1 = np.vstack((X_vector[:k*f],X_vector[(k+1)*f:])), X_vector[k*f:(k+1)*f]
        y_train1, y_test1 = np.hstack((y_vector[:k*f],y_vector[(k+1)*f:])), y_vector[k*f:(k+1)*f]
        print("Mean of X_train set: ",np.mean(X_train1))
        # X_train = (X_train1 - X_train1.mean(axis=0)) / (X_train1.std(axis=0) + epsilon)
        X_train=X_train1
        y_train=y_train1
        # y_train = (y_train1 - y_train1.mean(axis=0)) / (y_train1.std(axis=0) + epsilon)
        theta = np.zeros(X_train.shape[1])

        # print(theta)
        alpha=0.00000001
        iter=1000

        # Calling all functions:
        c=[]
        iterate=[]
        for i in range(iter):
            # print("Number of iterations: ", i+1)
            hyp_fn=hypothesis_func(X_train, theta)
            e=errors(hyp_fn, y_train)
            gradients=gradients_fn(X_train,e)
            theta=update_thetas(alpha, theta, gradients)
            # print("Updated thetas: ", theta)

            J=cost_fn(e)
            # print("Cost fn: ", J, "\n")
            iterate.append(i)
            c.append(J)

        print("Thetas: ",theta)
        # X_test = (X_test1 - X_train1.mean(axis=0)) / (X_train1.std(axis=0) + epsilon)
        X_test=X_test1
        y_test=y_test1
        # y_test = (y_test1 - y_train1.mean(axis=0)) / (y_train1.std(axis=0) + epsilon)
        test_hyp_fn = hypothesis_func(X_test, theta)
        # print(test_hyp_fn)
        test_errors = errors(test_hyp_fn, y_test)
        test_cost = cost_fn(test_errors)
        print(f"Final cost on test set: {test_cost}")
        cost_dict[k+1]=test_cost
        plt.plot(iterate, c)
        # plt.show()
        print(len(y_test))

        r2=r2_score( y_test, test_hyp_fn )
        print("r2 score: ",r2,"\n")
    print("Cost values of different folds: ",cost_dict)

if __name__ == '__main__':
    main()