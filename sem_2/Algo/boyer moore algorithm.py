# Boyer moore algorithm :

# Prepare look up tables:
# bad character table:
# T="ATCGTACGTGCACCGTGC"

def bad_char_table(P, E):
    table=[]
    for i in range(len(E)):
        t=[]
        for j in range(len(P)-1, -1, -1):
            if E[i]==P[j]:
                t.append("-")
            elif E[i]!=P[j] :
                temp = False
                for k in range(j-1 , -1, -1):
                    if E[i]==P[k]:
                        temp=True
                        t.append(j-k-1)
                        break
                if temp==False:
                    t.append(j)
        t.reverse()
        table.append(t)
    return table


from itertools import islice
def good_suffix_table():
    P=["N", "O", "Y", "O"]
    # P=["N","E","E","D","L","E"]
    # P=["A","N","P","A","N","M","A","N"]
    # P=["B","A","N","A","N","A"]
    suffixs={"":0}
    t=""
    tempo=""
    for i in range(len(P)-1,-1,-1):
        t=P[i]+t
        suffixs[t]=0
    print(suffixs)

    for i in islice(suffixs, 1, None):
        temp=False
        for j in range(len(P)-len(i)-1, 0, -1):
            # print("".join(P[j-len(i):j]))
            can= "".join(P[j-len(i):j])
            #......strong_good_suffix.......
            if i==can and P[j-1]!=P[len(i)-1] :
                suffixs[i]= len(P)-len(i)-j
                tempo=i
                temp = True
                break
        if not temp:
            suffixs[i] = len(P) - suffixs.get(tempo, len(P))
            # suffixs[i] = len(P)-suffixs[tempo]-1
    print(suffixs)

def main():
    # P=["T","C","G","C"]
    # E=["A", "T", "G", "C"]
    # P = ["B", "A", "B", "A", "A", "A", "B"]
    P = ["N", "E", "E", "D", "L", "E", "S"]
    E = ["N", "E", "D", "L", "S"]
    print(bad_char_table(P, E))
    print(good_suffix_table())
if __name__ == '__main__':
    main()