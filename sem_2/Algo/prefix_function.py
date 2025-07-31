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

def main():
    P = ["a", "b", "a", "b", "a", "c", "a"]
    # P = ["N", "E", "E", "D", "L", "E", "S"]
    print("Prefix function:", prefix_function(P))

if __name__ == '__main__':
    main()