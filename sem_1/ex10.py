# Write a program to concatenate two strings, S1 and S2.

def concatenate_str(s1,s2):
    # concatenating two strings:
    s=s1+s2
    # s=s1+" "+s2 ----> if space is needed can be added
    return s

def main():
    s1=str(input("Enter first string: "))
    s2=str(input("Enter second string: "))
    # To call the function and print the output
    print(concatenate_str(s1,s2))

if __name__=="__main__":
    main()