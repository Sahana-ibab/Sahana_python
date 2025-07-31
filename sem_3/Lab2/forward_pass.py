import numpy as np

def relu(z):
    return np.maximum(0, z)

def softmax_fn(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)

def perceptron(z):
    result = []
    for val in z:
        if val >= 0:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

def input_func():
    n = int(input("Enter the number of inputs: "))
    x = [float(input(f"Enter x{i+1}: ")) for i in range(n)]
    print("Entered inputs:", x)

    l = int(input("Enter the number of hidden layers: "))
    W, B, af = [], [], []
    a = input("DEFAULT: Using ReLU in hidden layers and Softmax in output layer, \nelse if you want to enter activation function manually, type 'yes': ")
    y = input("If you want to enter weights manually, type 'yes': ")

    for j in range(l):
        u = int(input(f"Enter the number of neurons in layer {j+1}: "))
        w = np.random.randint(0, 2, size=(u, n)).astype(float)
        b = np.random.randint(0, 2, size=u).astype(float)

        if a.lower() == "yes":
            choice = input("Enter ReLU, Softmax, or perceptron: ")
            if choice.lower() == "relu":
                af.append(relu)
            elif choice.lower() == "softmax":
                af.append(softmax_fn)
            else:
                af.append(perceptron)
        else:
            af.append(relu)

        if y.lower() == 'yes':
            for k in range(u):
                for i in range(n):
                    w[k][i] = float(input(f"Enter weight w{k+1}{i+1} for Layer {j+1}: "))
                b[k] = float(input(f"Enter bias for neuron {k+1} in Layer {j+1}: "))

        W.append(w)
        B.append(b)
        n = u

    O = int(input("Enter the number of neurons in the output layer: "))
    oa = input("Enter 'yes' if you want to use Act_fn other than Softmax in the output layer (ignored, Softmax will be used): ")
    if oa.lower() == "yes":
        choice = input("Enter ReLU or perceptron: ")
        if choice.lower() == "relu":
            Oa = relu
        elif choice.lower() == "perceptron":
            Oa = perceptron
        else:
            Oa = softmax_fn
    else:
        Oa = softmax_fn
    ow = np.random.randint(0, 2, size=(O, n)).astype(float)
    bo = np.random.randint(0, 2, size=O).astype(float)

    if y.lower() == "yes":
        for o in range(O):
            for h in range(n):
                ow[o][h] = float(input(f"Enter weight w{o+1}{h+1} for output neuron {o+1}: "))
            bo[o] = float(input(f"Enter bias for output neuron {o+1}: "))

    return x, W, B, ow, bo, af, Oa

def feed_forward(x, W, B, ow, bo, af, Oa):
    x = np.array(x)
    for i in range(len(W)):
        z = np.dot(W[i], x) + B[i]
        x = af[i](z)
    output = np.dot(ow, x) + bo
    return Oa(output)

def main():
    x, W, B, ow, bo, af, Oa = input_func()
    print("\n----- Feedforward Process -----\n")
    result = feed_forward(x, W, B, ow, bo, af, Oa)
    print("Output Probabilities:", result)

if __name__ == '__main__':
    main()
