def prefix_function(P):
    pi = [0] * len(P)
    k = 0
    for q in range(1, len(P)):
        while k > 0 and P[k] != P[q]:
            k = pi[k - 1]
        if P[k] == P[q]:
            k += 1
        pi[q] = k
    return pi

def kmp_matcher(T, P, pi):
    n = len(T)
    m = len(P)
    q = 0
    for i in range(n):
        while q > 0 and P[q] != T[i]:
            q = pi[q - 1]
        if P[q] == T[i]:
            q += 1
        if q == m:
            print(f"Pattern matched at shift {i - m + 1}")
            q = pi[q - 1]

def main():
    P = ["a", "b", "a", "b", "a", "c", "a"]
    T = "bacbababaabcbababacaba"
    print("Prefix function:", prefix_function(P))
    pi = prefix_function(P)
    print("Match found! ")
    kmp_matcher(T, P, pi)

if __name__ == '__main__':
    main()
