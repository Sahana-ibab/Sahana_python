# Relu function
import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0,z)

def der_relu(z):
    r = relu(z)
    dr = []
    for i in r:
        if i == 0:
            dr.append(0)
        else:
            dr.append(1)
    return dr

def main():
    z=np.arange(-8, 8)
    print(z)
    relu_func=relu(z)
    print(relu_func)
    plt.plot(z, relu_func)

    print(der_relu(relu_func))
    der_rel=der_relu(z)
    plt.plot(z,der_rel)
    plt.legend(["g(z)", "g'(z)"])
    plt.show()

if __name__=="__main__":
    main()
