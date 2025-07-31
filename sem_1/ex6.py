# Write a function to check if a given number, N, is prime or not
def prime_num(N):
    # initialization of local variable prime
    prime="True"
    # it iterates from 2to N-1
    for i in range (2,N):
        if N%i==0:
            prime="False"
        else:
            continue
    return prime

def main():
    N=int(input("Enter natural a number: "))
    # print ("The number is prime is: ", prime_num(N))
    # calling function
    prime=prime_num(N)
    # formating output
    if prime=="False":
        print ("The number %d is composite number." %N)
    else:
        print ("The number %d is prime number." %N)
if __name__ == '__main__':
    main()