# Write a function to find the  highest frequency character in a string, S.

def high_freq(s):
    c=s.count(s[0])
    m=s[0]
    n=1
    for i in range (1,len(s)):
        c1=s.count(s[i])
        if c1>c:
            m=s[i]
            n=c1
    return m, n

def main():
    s=str(input("Enter a string: "))
    # print(high_freq(s))
    m, n=high_freq(s)
    print("max occurred letter is: ", m)
    print("num of occurrences: ", n)
if __name__ == '__main__':
    main()