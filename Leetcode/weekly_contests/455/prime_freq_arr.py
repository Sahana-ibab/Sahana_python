# # Check if Any Element in an Array Has Prime Frequency:
#
# class Solution(object):
#     def isprime(self, count):
#         if count<2:
#             return False
#         for i in range(2, int((count**0.5)+1)):
#             if count%i==0:
#                 return False
#         return True
#
#     def checkPrimeFrequency(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: bool
#         """
#         for i in set(nums):
#             count = nums.count(i)
#             if self.isprime(count):
#                 return True
#         return False
#
# def main():
#     nums = [2,3,4,5,4,6,4]
#     sol=Solution()
#     result = sol.checkPrimeFrequency(nums)
#     print("Result: ", result)
#
# if __name__ == '__main__':
#     main()
#
#

from collections import Counter

# Check if Any Element in an Array Has Prime Frequency:

class Solution(object):
    def isprime(self, i):
        if i<2:
            return False
        for j in range(2, int((i**0.5)+1)):
            if i%j==0:
                return False
        return True

    def checkPrimeFrequency(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        count = Counter(nums)
        for i in count.values():
            if self.isprime(i):
                return True
        return False

def main():
    nums = [2,3,4,5,4,6,4]
    sol=Solution()
    result = sol.checkPrimeFrequency(nums)
    print("Result: ", result)

if __name__ == '__main__':
    main()


