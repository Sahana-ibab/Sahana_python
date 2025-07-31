import numpy as np
from numpy.linalg import eig

# 1. Check if a matrix is positive definite
def is_positive_definite(matrix):
    eigenvalues = eig(matrix)[0]
    print("Eigenvalues:", eigenvalues)
    return np.all(eigenvalues > 0)

# 2. Compute eigenvalues of Hessian at a given point
def hessian_eigenvalues_at_point(x, y):
    H = np.array([[12 * x ** 2, -1],
                  [-1, 2]])
    eigenvalues = eig(H)[0]
    print(f"\nHessian at ({x}, {y}):\n", H)
    print("Eigenvalues:", eigenvalues)
    return eigenvalues

# 3. Determine concavity of f(x, y) = x^3 + 2y^3 - xy
def concavity_fx(x, y):
    H = np.array([[6*x, -1], [-1, 12*y]])
    eigenvalues = eig(H)[0]
    print(f"\nPoint ({x},{y}) → Hessian:\n{H}")
    print("Eigenvalues:", eigenvalues)
    if np.all(eigenvalues > 0):
        print("Local Minimum")
    elif np.all(eigenvalues < 0):
        print("Local Maximum")
    else:
        print("Saddle Point")

# 4. f(x,y) = 4x + 2y - x^2 - 3y^2 → critical point and Hessian
def analyze_quadratic():
    # Gradient = [4 - 2x, 2 - 6y] → set to 0
    x_crit = 2
    y_crit = 1/3
    H = np.array([[-2, 0], [0, -6]])
    eigenvalues = eig(H)[0]
    print(f"\nCritical point: ({x_crit}, {y_crit})")
    print("Hessian:\n", H)
    print("Eigenvalues:", eigenvalues)
    if np.all(eigenvalues > 0):
        print("Local Minimum")
    elif np.all(eigenvalues < 0):
        print("Local Maximum")
    else:
        print("Saddle Point")

# Run all tasks
A = np.array([[9, -15], [-15, 21]])
print("1. Is A positive definite?")
print("Answer:", "Yes" if is_positive_definite(A) else "No")

print("\n2. Eigenvalues of Hessian at (3,1):")
hessian_eigenvalues_at_point(3, 1)

print("\n3. Concavity of f(x,y) = x^3 + 2y^3 - xy")
concavity_fx(0, 0)
concavity_fx(3, 3)
concavity_fx(3, -3)

print("\n4. Analyze f(x,y) = 4x + 2y - x^2 - 3y^2")
analyze_quadratic()
