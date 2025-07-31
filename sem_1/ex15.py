# Write a function to count the number of occurrences of a word, W, in a sentence, S.

def occur_word(s,w):
    # To split the sentence by spaces
    s=s.split(" ")
    c=0
    # To iterate through len of sentence and count the ocuurence
    for i in range (0,len(s)):
        if w==s[i]:
            c=c+1
        else:
            continue
    return c

def main():
    s=str(input("Enter a sentence: "))
    w=str(input("Enter the word to be checked: "))
    c=occur_word(s,w)
    print ("Number of occurences is : ", c)

if __name__=="__main__":
    main()