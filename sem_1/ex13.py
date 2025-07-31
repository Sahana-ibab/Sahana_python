# Write a function to replace all occurrences of a character, c,  with another character, d.

def replace_chr(s,c,d):
    l=list(s)
    for i in range (0, len(l)):
        if l[i]==c:
            l[i]=d
        else:
            continue
    ss =''.join(l)
    return ss

def main():
    s=str(input("Enter a string: "))
    c=str(input("Enter letter to be replaced: "))
    d=str(input("Enter letter to be replaced with: "))
    print("new string: ",replace_chr(s,c,d))

if __name__ == '__main__':
    main()