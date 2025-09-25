import numpy as np
from tanH_fn import tanH

def RNN_forward_pass(x, ht, wxh, whh, why):

    T = len(x)
    Y = []
    for i in range(T):
        h = np.matmul(whh, ht[i]) + np.matmul(wxh, x[i])
        # print(h)
        th = tanH(h)
        print(f"h{i+1}: ",th)
        ht.append(th)
        y = np.matmul(why, th)
        print(f"y{i+1}",y)


def main():
    x = [[1, 2], [-1, 1], [2, 3]]
    ht = [[0, 0, 0]]
    wxh = [[0.5, -0.3], [0.8, 0.2], [0.1, 0.4]]
    whh = [[0.1, 0.4, 0.0], [-0.2, 0.3, 0.2], [0.05, -0.1, 0.2]]
    why = [[1, -1, 0.5], [0.5, 0.5, -0.5]]
    RNN_forward_pass( x, ht, wxh, whh, why)


if __name__ == '__main__':
    main()

