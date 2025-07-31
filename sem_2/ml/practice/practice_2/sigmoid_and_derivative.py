import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))
# np.random.seed(42)
# z=np.random.randint(-100,100,size=10)
# plt.scatter(z,sigmoid(z),color='red')
# z_sorted=np.sort(z)
# sigmoid_sorted=np.sort(sigmoid(z))
# plt.plot(z_sorted,sigmoid_sorted,color='blue')
# plt.xlabel('z')
# plt.ylabel('sigmoid(z)')
# plt.title('sigmoid function')
# plt.show()

np.random.seed(64)
z=np.random.randint(-100,100,10)
z_sorted=np.sort(z)
derivative=z*(1-sigmoid(z))
der_sorted=np.sort(derivative)
plt.scatter(z,derivative,color='pink')
plt.plot(z_sorted,der_sorted,color='navy')
plt.xlabel('z')
plt.ylabel('sigmoid derivative')
plt.title('Derivative of sigmoid function')
plt.show()


