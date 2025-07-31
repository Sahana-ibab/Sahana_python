# Write print alternate characters of a string, S


def alt_str(word):
    # [::2] jumps by 2
    alt_word=word[::2]
    return alt_word

def main():
    word=str(input("Enter a word: "))
    print(alt_str(word))

if __name__ == '__main__':
    main()