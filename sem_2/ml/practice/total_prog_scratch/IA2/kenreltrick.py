import numpy as np

# Feature mapping function
def transform(x):
    return np.array([x[0] * x[0], x[0] * x[1], x[0] * x[2],
                     x[1] * x[0], x[1] * x[1], x[1] * x[2],
                     x[2] * x[0], x[2] * x[1], x[2] * x[2]])

# Input vectors
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Transform x and y using the feature mapping
phi_x = transform(x)
phi_y = transform(y)

# Compute the dot product in the higher-dimensional space
dot_product = np.dot(phi_x, phi_y)

# Print the result
print("Transformed x:", phi_x)
print("Transformed y:", phi_y)
print("Dot product in the higher-dimensional space:", dot_product)

# Kernel function
def kernel(x, y):
    return (x[0] * y[0])**2 + (x[0] * y[1]) * (x[1] * y[0]) + (x[0] * y[2]) * (x[2] * y[0]) + \
           (x[1] * y[0]) * (x[0] * y[1]) + (x[1] * y[1])**2 + (x[1] * y[2]) * (x[2] * y[1]) + \
           (x[2] * y[0]) * (x[0] * y[2]) + (x[2] * y[1]) * (x[1] * y[2]) + (x[2] * y[2])**2

# Compute the kernel function output
kernel_output = kernel(x, y)

# Print the result
print("Kernel function output (K(x, y)):", kernel_output)
