# Data normalization - scale the values between 0 and 1. Implement code from scratch.
# import numpy as np
#
# def data_normalization(data):
#     min=float('inf')
#     max=float('-inf')
#     norm_data=[]
#     for i in data:
#         if i<min:
#             min=i
#         if i>max:
#             max=i
#     for j in data:
#         norm_data.append((j-min)/(max-min))
#     return norm_data
#
# def main():
#     data= [1,2,2,3,4,5,6,7]
#
#     data_norm=data_normalization(data)
#     print(data_norm)
#     print("Mean of normalization data: ", np.mean(data_norm))
#     print("Standard deviation of normalization data: ", np.std(data_norm))
#
# if __name__ == '__main__':
#     main()


import numpy as np
import pandas as pd
def data_normalization(data):
    min=float('inf')
    max=float('-inf')
    norm_data=[]
    for i in data:
        if i<min:
            min=i
        if i>max:
            max=i
    for j in data:
        norm_data.append((j-min)/(max-min))
    return norm_data

def main():
    data= pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
    X=data[["age"]]

    data_norm=data_normalization(X)
    print(data_norm)
    print("Mean of normalization data: ", np.mean(data_norm))
    print("Standard deviation of normalization data: ", np.std(data_norm))

if __name__ == '__main__':
    main()


