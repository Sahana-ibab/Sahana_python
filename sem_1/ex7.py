# Write a function to print individual digits of a number, N.

def ind_dig(num):
    dig=""
    while num>0:
        dig = str(dig) + str( num % 10 ) + str("  ")
        num=num//10
    return dig

def main():
    num=int(input("Enter a number to get its digits: "))
    print ("The digits are: ", ind_dig(num))
#
if __name__=="__main__":
    main()