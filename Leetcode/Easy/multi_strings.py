# Multiply Strings
# You are given two strings num1 and num2 that represent non-negative integers.
# Return the product of num1 and num2 in the form of a string.
# Assume that neither num1 nor num2 contain any leading zero, unless they are the number 0 itself.
# Note: You can not use any built-in library to convert the inputs directly into integers.

class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        res = ""
        for i in num2[::-1]:
            res +=i*int(num1)*10
        return res


def main():
    num1 = "12"
    num2 = "13"

    sol = Solution()
    print(sol.multiply(num1, num2))



if __name__ == '__main__':
    main()
