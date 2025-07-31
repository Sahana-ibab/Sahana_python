import matplotlib.pyplot as plt
import math as m
def plot_graph(a,b,c):
    x=[-5,-4,-3,-2,-1,0,1,2,3,4,5]
    y=[(a*i*i)+(b*i)+c for i in x]
    # y = [-(i*i) for i in x]

    #.....for 2nd question...a)
    # y = [(i ** 2) + 1 for i in x]

    # .....for 2nd question...b)
    # x = [i * 0.1 for i in range(-12, 33)]
    # y = [(m.sqrt(4-((i-1) ** 2) + 1)) for i in x]
    plt.plot(x,y)
    plt.show()


def main():
    a = int(input("a: "))
    b = int(input("b: "))
    c = int(input("c: "))

    plot_graph(a,b,c)


if __name__ == '__main__':
    main()