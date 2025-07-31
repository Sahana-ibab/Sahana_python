# A non-cyclical number is an integer defined by the following algorithm:
#
#     Given a positive integer, replace it with the sum of the squares of its digits.
#     Repeat the above step until the number equals 1, or it loops infinitely in a cycle which does not include 1.
#     If it stops at 1, then the number is a non-cyclical number.
#
# Given a positive integer n, return true if it is a non-cyclical number, otherwise return false.


class Solution:
    def cal_squares_digits(self, n):

        output = 0
        while n > 0:
            digit = (n % 10) ** 2
            output += digit
            n = n//10
        return output


    # def isHappy(self, n: int) -> bool:
    #
    #     visit = set()
    #
    #     while n not in visit:
    #         visit.add(n)
    #         n = self.cal_squares_digits(n)
    #
    #         if n == 1:
    #             return True
    #     return False

    def isHappy(self, n: int) -> bool:
        slow, fast = n, self.cal_squares_digits(n)

        while slow != fast:
            fast = self.cal_squares_digits(self.cal_squares_digits(fast))
            slow = self.cal_squares_digits(slow)
            if fast == 1:
                return True
        return False


def main():
    n = 19
    sol = Solution()
    if sol.isHappy(n):
        print(f"{n} is cyclic.")
    else:
        print(f"{n} is non-cyclic.")



if __name__ == '__main__':
    main()