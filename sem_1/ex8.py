# Write a function to print the first half of a string, S.
def half_str(word):
    # get half-length of string
    half_len=len(word)//2     # // ---> to convert float to int
    # getting half-string by iterating from 0 to half-len to variable
    half_word=str(word[0:half_len])
    # returning half string
    return half_word

def main():
    word=str(input("Enter a string: "))
    # printing the output
    print(half_str(word))

if __name__ == '__main__':
    main()