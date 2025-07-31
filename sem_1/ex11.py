# Write a function to find the first occurrence of a character, c, in a string S.
def occurrence_char(s,c):
    # iterating and searching for occurrence
    for i in range (0,len(s)):
        if s[i]==c:
            break
        else:
            continue
    return i

def main():
    s=str(input("Enter a string: "))
    c=str(input("Enter the character to be checked: "))
    #  index starts with 0
    print("The string occurs at index: ",occurrence_char(s,c))

if __name__ == '__main__':
    main()