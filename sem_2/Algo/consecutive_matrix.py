import re
from itertools import permutations

def row_combinations(M):
    row_permutations = list(permutations(M))
    return row_permutations

def consecutive_property(row_comb):
    for M in row_comb:
        temp=True
        # print(M)
        for i in zip(*M):
            s = "".join(map(str, i))
            # print(s)
            p = "101"
            if re.search(p, s):
                temp=False
                break
        if temp:
            return "Matrix is consecutive"
    return "Matrix is not consecutive."
def main():
    M = [[1, 1, 0], [0, 1, 1], [1, 0, 0]]
    # M = [[0, 1, 0], [1, 0, 1], [1, 0, 0]]
    M2 = [
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]]

    row_comb=row_combinations(M2)
    print(consecutive_property(row_comb))
if __name__ == '__main__':
    main()


