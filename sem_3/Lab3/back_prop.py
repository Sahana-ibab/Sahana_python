import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax_fn(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)

def perceptron(z):
    return np.array([1 if val >= 0 else 0 for val in z])

def perceptron_deriv(z):
    return np.zeros_like(z)

def forward_pass(x, W, B, af, ow, bo):
    activations = [np.array(x)]
    zs = []
    a = np.array(x)

    for i in range(len(W)):
        z = np.dot(W[i], a) + B[i]
        zs.append(z)
        a = af[i](z)
        activations.append(a)

    z_out = np.dot(ow, a) + bo
    zs.append(z_out)
    a_out = softmax_fn(z_out)
    activations.append(a_out)

    return activations, zs

def back_prop(activations, zs, y_true, W, B, ow, bo, af_deriv, lr):
    y_pred = activations[-1]
    delta_out = y_pred - y_true

    ow -= lr * np.outer(delta_out, activations[-2])
    bo -= lr * delta_out

    delta = delta_out
    for i in reversed(range(len(W))):
        delta = np.dot(ow.T, delta) * af_deriv[i](zs[i])
        W[i] -= lr * np.outer(delta, activations[i])
        B[i] -= lr * delta

    return W, B, ow, bo

def main():
    x = [1.0, 0.5]
    W = [np.array([[0.2, 0.4], [0.5, 0.3]])]
    B = [np.array([0.1, 0.2])]
    ow = np.array([[0.6, 0.9], [0.1, 0.8]])
    bo = np.array([0.05, 0.05])
    af = [relu]
    af_deriv = [relu_deriv]
    y_true = np.array([1, 0])
    lr = 0.1

    print("\n----- Feedforward Process -----\n")
    activations, zs = forward_pass(x, W, B, af, ow, bo)
    print("Output Probabilities:", activations[-1])

    print("\n----- Backpropagation -----\n")
    W, B, ow, bo = back_prop(activations, zs, y_true, W, B, ow, bo, af_deriv, lr)
    print("Updated Weights of hidden layers:", W)
    print("Updated Bias of hidden layers:", B)
    print("Updated Weights of output layer:", ow)
    print("Updated Bias of output layer:", bo)

if __name__ == '__main__':
    main()
