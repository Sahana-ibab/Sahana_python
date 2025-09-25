#
#
# def max_pooling(matrix, dim):
#     mat = []
#     for i in range(len(matrix) - dim + 1):
#         m = []
#         for j in range(len(matrix[i]) - dim + 1):
#             temp = 0
#             for k in range(dim):
#                 for l in range(dim):
#                    if temp < matrix[i + k][j + l]:
#                        temp = matrix[i + k][j + l]
#             m.append(temp)
#         mat.append(m)
#     return mat
#
#
# def convolution_operation(matrix, filter, p, s):
#     n, m = len(matrix), len(matrix[0])
#     f_n, f_m = len(filter), len(filter[0])
#     out_rows = (n + 2 * p - f_n) // s + 1
#     out_cols = (m + 2 * p - f_m) // s + 1
#     mat=[]
#     for i in range(0, out_rows):
#         i_s = i *s
#         mt = []
#         for j in range(0, out_cols):
#             j_s = j * s
#             temp = 0
#             for k in range(f_n):
#                 for l in range(f_m):
#                     if 0 <= i_s+k < n and 0<= j_s+l < m:
#                         t= matrix[i_s + k][j_s + l]
#                     else:
#                         t=0
#                     temp += t * filter[k][l]
#             mt.append(temp)
#         mat.append(mt)
#     return mat
#
#
# def main():
#     matrix = [[3, 0, 1, 2, 7, 4],
#               [1, 5, 8, 9, 3, 1],
#               [2, 7, 2, 5, 1, 3],
#               [0, 1, 3, 1, 7 ,8],
#               [4, 2, 1, 6, 2, 8],
#               [2, 4, 5, 2, 3, 9]]
#     filter = [[1, 0, -1] ,[1, 0, -1] ,[1, 0, -1]]
#     p=1
#     s=2
#     print(matrix)
#     print(convolution_operation(matrix, filter, p,s))
#     print(max_pooling(matrix, 3))
#
#
#
# if __name__ == '__main__':
#     main()
#
#


#
# def max_pooling(matrix, dim):
#     fin =[]
#     for c in range(len(matrix)):
#         mat = []
#         for i in range(len(matrix[c]) - dim + 1):
#             m = []
#             for j in range(len(matrix[c][i]) - dim + 1):
#                 temp = 0
#                 for k in range(dim):
#                     for l in range(dim):
#                        if temp < matrix[c][i + k][j + l]:
#                            temp = matrix[c][i + k][j + l]
#                 m.append(temp)
#             mat.append(m)
#         fin.append(mat)
#     return fin
#
#
# def convolution_operation(matrix, filter, p, s):
#     n, m = len(matrix[0]), len(matrix[0][0])
#     f_n, f_m = len(filter[0]), len(filter[0][0])
#     out_rows = (n + 2 * p - f_n) // s + 1
#     out_cols = (m + 2 * p - f_m) // s + 1
#     fin=[]
#     for c in range(len(matrix)):
#         mat=[]
#         for i in range(0, out_rows):
#             i_s = i *s
#             mt = []
#             for j in range(0, out_cols):
#                 j_s = j * s
#                 temp = 0
#                 for k in range(f_n):
#                     for l in range(f_m):
#                         if 0 <= i_s+k < n and 0<= j_s+l < m:
#                             t= matrix[c][i_s + k][j_s + l]
#                         else:
#                             t=0
#                         temp += t * filter[c][k][l]
#                 mt.append(temp)
#             mat.append(mt)
#         fin.append(mat)
#     return fin
#
#
# def main():
#     matrix = [[[3, 0, 1, 2, 7, 4],
#               [1, 5, 8, 9, 3, 1],
#               [2, 7, 2, 5, 1, 3],
#               [0, 1, 3, 1, 7 ,8],
#               [4, 2, 1, 6, 2, 8],
#               [2, 4, 5, 2, 3, 9]],
#               [[3, 0, 1, 2, 7, 4],
#               [1, 5, 8, 9, 3, 1],
#               [2, 7, 2, 5, 1, 3],
#               [0, 1, 3, 1, 7 ,8],
#               [4, 2, 1, 6, 2, 8],
#               [2, 4, 5, 2, 3, 9]],
#               [[3, 0, 1, 2, 7, 4],
#               [1, 5, 8, 9, 3, 1],
#               [2, 7, 2, 5, 1, 3],
#               [0, 1, 3, 1, 7, 8],
#               [4, 2, 1, 6, 2, 8],
#               [2, 4, 5, 2, 3, 9]]]
#     filter = [[[1, 0, -1] ,[1, 0, -1] ,[1, 0, -1]],
#               [[1, 0, -1] ,[1, 0, -1] ,[1, 0, -1]],
#               [[1, 0, -1] ,[1, 0, -1] ,[1, 0, -1]]]
#     p=1
#     s=2
#     # print(matrix)
#     print(convolution_operation(matrix, filter, p,s))
#     print(max_pooling(matrix, 3))
#
#
#
# if __name__ == '__main__':
#     main()

import random
import numpy as np


def max_pooling(matrix, dim, p, s):
    n, m = len(matrix[0]), len(matrix[0][0])
    out_rows = ((n + 2 * p - dim) // s) + 1
    out_cols = ((m + 2 * p - dim) // s) + 1
    fin =[]
    for c in range(len(matrix)):
        mat = []
        for i in range(out_rows):
            i_s = i * s
            mt = []
            for j in range(out_cols):
                j_s = j * s
                temp = float('-inf')
                for k in range(dim):
                    for l in range(dim):
                        if 0 <= i_s + k < n and 0 <= j_s + l < m:
                            val = matrix[c][i_s + k][j_s + l]
                        else:
                            val = 0
                        if val > temp:
                            temp = val
                mt.append(temp)
            mat.append(mt)
        fin.append(mat)
    return fin


def convolution_operation(matrix, filters, p, s):
    n, m = len(matrix[0]), len(matrix[0][0])
    f_n, f_m = len(filters[0][0]), len(filters[0][0][0])
    out_rows = ((n + 2 * p - f_n) // s) + 1
    out_cols = ((m + 2 * p - f_m) // s) + 1
    final=[]
    for f in range(len(filters)):
        fin=np.zeros((out_rows,out_cols))
        for c in range(len(matrix)):
            mat=[]
            for i in range(out_rows):
                i_s = i *s
                mt = []
                for j in range(out_cols):
                    j_s = j * s
                    temp = 0
                    for k in range(f_n):
                        for l in range(f_m):
                            if 0 <= i_s+k < n and 0<= j_s+l < m:
                                t= matrix[c][i_s + k][j_s + l]
                            else:
                                t=0
                            temp += t * filters[f][c][k][l]
                    mt.append(temp)
                mat.append(mt)
            fin = np.add(fin, mat)
        final.append(fin)
    return final


def main():

    matrix = [[[random.randint(0, 255) for _ in range(32)] for _ in range(32)] for _ in range(4)]
    filters = [[[[random.randint(0, 9) for _ in range(3)] for _ in range(3)] for _ in range(4)] for _ in range(6)]

    p=1
    s=2
    print("IMAGE: ",matrix)
    conv_out= convolution_operation(matrix, filters, p, s)
    print("CONVOLUTION OUTPUT: ", conv_out)
    print("MAX-POOLING OUTPUT: ",max_pooling(matrix, 3, p , s))
    print("Dimension of CONV-output: ", np.shape(conv_out))
if __name__ == '__main__':
    main()
