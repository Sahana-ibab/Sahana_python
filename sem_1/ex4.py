# Write a function to convert the binary number B into decimal.
def bin_dec(num):
    d=0
    n=0
    # To check whether input number is binary or not:
    dig_count=str(num).count("0") + str(num).count("1")
    if dig_count==len(str(num)):
        while num>0:
            d=d+((num%2)*(2**n))
            num=num//10
            n=n+1
        return d
    else:
        return "Invalid output!"

def main():
    num=int(input("Enter a binary number: "))
    print(bin_dec(num))

if __name__ == '__main__':
    main()
