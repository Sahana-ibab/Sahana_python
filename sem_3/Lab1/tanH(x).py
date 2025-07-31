# TanH function
import numpy as np
import matplotlib.pyplot as plt

def tanH(z):
    return ((np.exp(z))-(np.exp(-z)))/((np.exp(z))+(np.exp(-z)))

def der_tanH(z):
    der=(1-(((np.exp(z))-(np.exp(-z)))/((np.exp(z))+(np.exp(-z))))**2)
    return der
def main():
    z=np.arange(-8, 8)
    print(z)
    tanH_func=tanH(z)
    print(tanH_func)
    plt.plot(z, tanH_func)

    print(der_tanH(tanH_func))
    der_tan=der_tanH(z)
    plt.plot(z,der_tan)
    plt.legend(["g(z)", "g'(z)"])
    plt.show()

    avg = np.sum(tanH_func) / len(tanH_func)
    print("Average: ", avg)
    if np.isclose(0, avg):
        print("Function is Zero-centered")
    else:
        print("Function is not Zero-centered")

if __name__=="__main__":
    main()
