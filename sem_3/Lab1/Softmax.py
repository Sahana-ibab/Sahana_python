# Softmax function
import numpy as np
import matplotlib.pyplot as plt

def softmax_func(z):
    return np.exp(z)/(np.sum(np.exp(z)))

def der_soft(sof_func):
    leng = len(sof_func)
    der= np.zeros((leng, leng))
    for i in range(leng):
        for j in range(leng):
            if i==j:
                der[i][j] = sof_func[i]*(1-sof_func[i])
            else:
                der[i][j] = -sof_func[i]*sof_func[j]
    return der

def main():
    z=np.arange(-8, 8)
    print(z)
    sof_func=softmax_func(z)
    print(sof_func)
    plt.plot(z, sof_func)

    print(der_soft(sof_func))
    der_softmax=der_soft(sof_func)
    plt.plot(z,der_softmax)
    plt.legend(["g(z)", "g'(z)"])
    plt.show()

if __name__=="__main__":
    main()
