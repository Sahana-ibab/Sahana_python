import numpy as np

# polynomial kernel--- p dim (where d << p):
def polynomial_kernel(a, b):
    return (a[0] * b[0])**2 + 2 * a[0] * b[0] * a[1] * b[1] + (a[1] * b[1])**2

# Kernel trick function : outputs the dot-product
def kernel_trick_equiv(a, b):
    return (np.dot(a, b)) ** 2

x1 = [1, 1, 2, 3, 6, 9, 13, 18, 3, 6, 6, 9, 10, 11, 12, 16]
x2 = [13, 18, 9, 6, 3, 2, 1, 1, 15, 6, 11, 5, 10, 5, 6, 3]

# Looping over pairs and applying both kernels:
for i in range(len(x1)):
    a = [x1[i], x2[i]]   # comparing with itself
    b = [x1[i], x2[i]]
    print(f"Point: ({x1[i]}, {x2[i]})")
    print(f"  Kernel: {polynomial_kernel(a, b)}, Trick: {kernel_trick_equiv(a, b)}\n")
