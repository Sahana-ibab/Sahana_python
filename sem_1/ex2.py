#Write a function to convert the decimal number D into binary.
def dec_bin(D):
    if D==1 or D==0:
        return D
    elif D<0:
        return "invalid input"
    else:
        B=""
        while D>0:
            B=str( D%2 )+B
            D=D//2
        return B
def main():
    D=int(input("Enter a Decimal number: "))
    B=dec_bin(D)
    print (dec_bin(D))
if __name__ == '__main__':
    main()
