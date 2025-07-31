import numpy as np

def check_quadratic_form(A):
    # List of test vectors
    test_vectors = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([-1, 1]),
        np.array([1, -1]),
        np.array([5, 3])
    ]

    signs = []

    for x in test_vectors:
        val = x.T @ A @ x
        print(f"x = {x}, x^T A x = {val}")
        signs.append(np.sign(val))

    unique_signs = set(signs)

    if unique_signs == {1}:
        print("Matrix is positive definite.")
    elif unique_signs == {-1}:
        print("Matrix is negative definite.")
    elif unique_signs <= {0} and 0 in unique_signs:
        print("Matrix is negative semi-definite.")
    elif unique_signs <= {1} and 0 in unique_signs:
        print("Matrix is positive semi-definite.")
    elif 1 in unique_signs and -1 in unique_signs:
        print("Matrix is indefinite.")
    else:
        print("Could not determine definiteness.")

def main():
    A = np.array([[9, -15], [-15, 21]])
    check_quadratic_form(A)

if __name__ == '__main__':
    main()
