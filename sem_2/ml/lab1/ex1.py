#At.A:
def matrix_multi(A,AT):
    M = [[0] * len(AT[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(AT[0])):
            for k in range(len(A[0])):
                M[i][j] += A[i][k] * AT[k][j]
    return M

def transpose_mat(A):
    AT = [[0 for _ in range(len(A))] for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            AT[j][i]=A[i][j]
    return AT

def main():
    A=[[1,2,3],[4,5,6]]
    print("A: ",A)
    AT=transpose_mat(A)
    print("Transpose of A: ", AT)
    print("A*At: ",matrix_multi(A,AT))

if __name__ == '__main__':
    main()