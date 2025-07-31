#To get sum of squares of n numbers:
def sum_squares_n(num):
    s=0
    for i in range (1,num+1):
        s=s+(i*i)
        return s
def main():
    num=int(input("Enter value of N: "))
    ans=sum_squares_n(num)
    print (ans)
if __name__ == '__main__':
    main()
