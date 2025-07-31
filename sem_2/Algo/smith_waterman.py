import pandas as pd

def create_scoring_matrix(alphabet, mismatch_penalty):

    matrix = pd.DataFrame(mismatch_penalty, index=alphabet, columns=alphabet)

    for base in alphabet:
        matrix.loc[base, base] = 3

    return matrix

def smith_waterman_algo(scoring_matrix, G, s1, s2):
    m=len(s1)
    n=len(s2)
    F=[[0]* (n+1) for _ in range(m+1)]
    # print(F)
    for j in range(1, n+1):
        F[0][j]=0
    for i in range(1, m+1):
        F[i][0]= 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            F[i][j]= max(F[i-1][j-1]+ scoring_matrix.loc[s1[i-1],s2[j-1]], F[i-1][j]+G, F[i][j-1]+G, 0)

    return F

def print_smith_waterman_table(F):
    F = [[int(cell) for cell in row] for row in F]
    for row in F:
        print(row)

def trace_smith_waterman_table(F, s1, s2, scoring_matrix, G):
    LCS=""
    # i = len(F)-1
    # j= len(F[0])-1
    i, j = len(s1), len(s2)
    d=0
    while i > 0 or j > 0:
        if F[i][j] == F[i-1][j-1] + scoring_matrix.loc[s1[i-1],s2[j-1]]:
            if s1[i-1] == s2[j-1]:
                LCS = s1[i-1] + LCS
            i-=1
            j-=1
        elif F[i][j] == F[i-1][j] + G:
            i-=1
        else:
            j-=1
        d+=1
    return LCS


def main():
    s2="CAGCTA"
    s1="CACATA"
    # s1="CCATGCGA"
    # s2="ACGTTGCA"
    s1='TGTTACGG'
    s2='GGTTGACTA'

    alphabet = ['A', 'C', 'G', 'T']
    G = -2
    mismatch_penalty = -3
    scoring_matrix=create_scoring_matrix(alphabet, mismatch_penalty)
    F=smith_waterman_algo(scoring_matrix, G, s1, s2)
    print("Needleman-Wunsch Table: ")
    print_smith_waterman_table(F)
    print("LCS: ",trace_smith_waterman_table(F, s1, s2, scoring_matrix, G))

if __name__ == '__main__':
    main()