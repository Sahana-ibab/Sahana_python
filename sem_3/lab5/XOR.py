import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

def softmax_func(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)

def feed_forward(x, W, B, ow, bo):
    x = np.array(x)
    for i in range(len(W)):
        z = np.dot(W[i], x) + B[i]
        x = relu(z)
    output = np.dot(ow, x) + bo
    return relu(output), x

def main():

    X = [[0,0], [0,1], [1,0], [1,1]]

    W = [np.array([[1, 1], [1, 1]])]
    B = [np.array([0, -1])]

    ow = np.array([[1, -2]])
    bo = np.array([0])

    print("\n----- Feedforward XOR Network -----")
    print("DEFAULT: Using ReLU in hidden layers and in output layer.\n")

    p=[]
    p2=[]
    for x in X:
        result, output_h1 = feed_forward(x, W, B, ow, bo)
        print(f"Input: {x}, Output Probabilities: {result} ")
        p.append(output_h1[0])
        p2.append(output_h1[1])
        # print(p)
    plt.grid(True)
    plt.scatter(p, p2)
    plt.show()


if __name__ == '__main__':
    main()
