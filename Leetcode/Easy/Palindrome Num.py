# Without converting it to string:

class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """

        if x < 0 or (x % 10 == 0 and x != 0):   # If it is negative or ending with 0's
            return False
        reversed_half = 0
        while x > reversed_half:
            reversed_half = 10 * reversed_half + x%10
            x = x//10
        return x  == reversed_half or x == reversed_half // 10

# Note:
# /  -> floating point
# // -> floor
# %  -> Modules (remainder)

def main():
    x = 121
    # x = -121
    # x = 120
    sol = Solution()
    result = sol.isPalindrome(x)
    print(result)

if __name__ == '__main__':
    main()


