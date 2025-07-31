# Write a function to trim leading whitespace characters from a string, S.

def whitespace_strip(s):
    s=s.strip()
    return s

def main():
    s=str(input("Enter a string with spaces: "))
    print(whitespace_strip(s))

if __name__=="__main__":
    main()