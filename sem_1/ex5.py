# Write a function that computes power - raise base to the n-th power. Eg. power(2, 5).
# Here base is 2 and n-th power is 5.
def power_base(i,j):
    # here inbuilt function pow is used
    ans=pow(i,j)
    return ans

def main():
    #user inputs
    i=int(input("Enter base value: "))
    j=int(input("Enter power value: "))
    #calling function
    ans=power_base(i,j)
    #output
    print (ans)

if __name__ == '__main__':
    main()
