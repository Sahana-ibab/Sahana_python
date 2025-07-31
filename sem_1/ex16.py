# Write a function to check if two strings are anagrams of each other .
# Eg. listen and silent are anagrams, gram and arm are not anagrams


def anagrams(s1,s2):
    if len(s1)==len(s2):
        for i in range (0,len(s1)):
            prst="false"
            for j in range (0,len(s2)):
                if s1[i]==s2[j]:
                    prst="true"
                    break
                else:
                    continue
            if prst=="false":
                break
            else :
                continue
    return prst

def main():
    s1=str(input("Enter string 1: "))
    s2=str(input("Enter string 2: "))
    prst=anagrams(s1,s2)
    if prst=="true":
        print("They are anagrams.")
    else:
        print("They are not anagrams.")

if __name__=="__main__":
    main()