# Sigmoid function
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_func(z):
    return 1/(1+np.exp(-z))

def der_sig(z):
    der=((1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z)))))
    return der
def main():
    z=np.arange(-8, 8)
    print(z)
    sig_func=sigmoid_func(z)
    print(sig_func)
    plt.plot(z, sig_func)

    print(der_sig(sig_func))
    der_sigm=der_sig(z)
    plt.plot(z,der_sigm)
    plt.legend(["g(z)", "g'(z)"])
    plt.show()

    avg=np.sum(sig_func)/len(sig_func)
    print("Average: ", avg)
    if avg==0:
        print("Function is Zero-centered")
    else:
        print("Function is not Zero-centered")

if __name__=="__main__":
    main()
