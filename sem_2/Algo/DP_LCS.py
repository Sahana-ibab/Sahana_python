#Least common Subsequence:

def LCS_Table(s1, s2):
    s1l=list(s1)
    s2l=list(s2)
    # s1l.append(" ")
    # s2l.append(" ")
    # print(s1l,s2l)
    m = len(s1l)
    n = len(s2l)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    # print(L)
    for i in range( m-1, -1, -1):
        for j in range( n-1, -1, -1):
            if s1l[i]=="" or s2l[j]=="":
                L[i][j]= 0
            elif s1l[i]==s2l[j]:
                L[i][j]= 1 + (L[i+1][j+1])
            else:
                if L[i+1][j] > L[i][j+1]:
                    L[i][j]= L[i+1][j]
                else:
                    L[i][j] = L[i][j+1]
    return L

def LCS_trace(LCS_table, s1, s2):
    LCS=""
    i=0
    j=0
    def LCS_length(LCS_table):
        return LCS_table[0][0]

    while i<len(s1):
        while j < len(s2):
            if s1[i]==s2[j]:
                LCS += s1[i]
                i=i+1
                j=j+1
            else:
                if LCS_table[i+1][j] > LCS_table[i][j+1]:
                    i=i+1
                else:
                    j=j+1
    return LCS, LCS_length(LCS_table)

def edit_dist(s1, s2, LCS_len):
    return len(s1)+len(s2)-(2*LCS_len)

def main():
    # s2="ACGTCATCA"
    # s1="TAGTGTCA"
    s1="bd "
    s2="abcd"

    LSC_table=LCS_Table(s1, s2)
    for row in LSC_table:
        print(row)

    LCS, LCS_len=LCS_trace(LSC_table, s1, s2)
    print("LCS: ",LCS)
    print("Length of LCS: ", LCS_len)
    print("Edit distance: ", edit_dist(s1, s2, LCS_len))

if __name__=="__main__":
    main()




















