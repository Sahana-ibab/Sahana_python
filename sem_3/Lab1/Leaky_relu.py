# LeakyRelu function
import numpy as np
import matplotlib.pyplot as plt

def l_relu(z, alpha):
    return np.maximum(alpha*z,z)

def der_relu(z, alpha):
    # r= l_relu(z, alpha)
    # dr = []
    # for i in r:
    #     if i<=0:
    #         dr.append(alpha)
    #     else:
    #         dr.append(1)
    # return dr

    return l_relu(z, alpha)/z

def main():
    z=np.arange(-8, 8)
    alpha=0.01
    print(z)
    relu_func=l_relu(z, alpha)
    print(relu_func)
    plt.plot(z, relu_func)

    print(der_relu(relu_func, alpha))
    der_rel=der_relu(z, alpha)
    plt.plot(z,der_rel)
    plt.legend(["g(z)", "g'(z)"])
    plt.show()

if __name__=="__main__":
    main()
