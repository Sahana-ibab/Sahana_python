# no Gaps are considered:

def Hamming_dist(s1, s2):
    count = 0
    m=len(s1)-1
    n=len(s2)-1
    i=0
    while i<=m and i<=n:
       if s1[i]!=s2[i]:
           count += 1
       i += 1
    count += abs(m-n)
    return count

def main():
    # s1="ACGTCATCA"
    # s2="TAGTGTCA"
    s1="10111"
    s2="11001"

    print("Hamming distance: ",Hamming_dist(s1, s2))

if __name__ == '__main__':
    main()