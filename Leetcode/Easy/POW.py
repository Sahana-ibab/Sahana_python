# Pow(x, n) is a mathematical function to calculate the value of x raised to the power of n
# (i.e., x^n). Given a floating-point value x and an integer value n, implement the myPow(x, n)
# function, which calculates x raised to the power n.
# You may not use any built-in library functions.

class Solution:
    def myPow(self, x: float, n: int) -> float:
        def helper(x, n):

            if n == 0:
                return 1
            if x == 0:
                return 0
            sqr = helper(x*x, n//2)
            return x*sqr if n%2 == 1 else sqr

        sqr = helper(x, abs(n))
        return sqr if n >= 0 else 1/sqr

def main():
    x = 2
    n = 6
    # n = -6
    sol = Solution()
    print(sol.myPow(x, n))

if __name__ == '__main__':
    main()