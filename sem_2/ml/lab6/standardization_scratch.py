# Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.
import math as mt
import numpy as np
def data_standardization(data):
    std_data=[]
    def mean(x):
        m = 0
        for i in x:
            m += i
        m = m / len(x)
        return m

    def standard_deviation(data,m):
        V = 0
        for i in data:
            V += (i - m) ** 2
        V = V / (len(data) - 1)
        SD = mt.sqrt(V)
        return SD

    m=mean(data)
    sd=standard_deviation(data,m)

    for j in data:
        std_data.append((j-m)/sd)
    return std_data

def main():
    data= [1,2,2,3,4,5,6,7]
    print(data_standardization(data))
    data_std=data_standardization(data)
    print("Mean of standardized data: ",np.mean(data_std))
    print("Standard deviation of standardized data: ",np.std(data_std))
if __name__ == '__main__':
    main()

