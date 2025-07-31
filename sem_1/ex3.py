from ex2 import dec_bin

def num_of_1s(B):
    C=int(B)
    C1=0
    C2=0
    while C>0:
        if C%2==1:
            C1=C1+1
        else:
            C2=C2+1
        C=C//10
    return C1,C2

def main():
    num=int(input("Enter a decimal number: "))
    B=dec_bin(num)
    ans=num_of_1s(B)
    print (ans)
if __name__ == '__main__':
    main()


