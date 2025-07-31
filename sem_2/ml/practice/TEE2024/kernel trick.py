# 2. Let x = [X1, X2, X3], and y = [y1, y2, y3]. Implement a feature mapping function, $(x), called Transform(x),
# given by $(x) = (X1X1, X1X2, X1X3, X2X1, X2X2, X2X3, X3X1, X3X2, X3X3).  a. Let x = [1, 2, 3], y = [4, 5, 6].
# Use the above "Transform" function to transform these vectors to a higher dimension and compute the dot product
# in a higher dimension. Print the value. b. Implement a kernel, K(x,y) = (< x,y >)².
# Apply this kernel function and evaluate the output for the same x and y vectors.
# Show that the result is the same in both scenarios demonstrating the power of a kernel trick.


def transform(x):
    phi=[]
    for i in range(0, len(x)):
        for j in range(0, len(x)):
            phi.append( x[i] * x[j] )

    return phi

def dot_product(phi_X, phi_Y):
    P = 0
    for i in range(0, len(phi_X)):
        P+=phi_X[i]*phi_Y[i]
    return P

def kernel(X, Y):
    K = 0
    for i in range(0, len(X)):
        s = X[i] * Y[i]
        K += s
    K = K**2
    return K

def main():
    X = [1, 2, 3]
    Y = [4, 5, 6]
    phi_X = transform(X)
    phi_Y = transform(Y)
    dot_prod = dot_product(phi_X, phi_Y)
    print(dot_prod)
    print(kernel(X, Y))

if __name__ == '__main__':
    main()

