from typing import List

class Solution:
    def hasDuplicate(self, nums: List[int]) -> bool:
            if len(nums) == len(set(nums)):
                return False
            else:
                return True

def main():
    A=[2,3,4,5,6,2]

    sol = Solution()

    if sol.hasDuplicate(A):
        print("The array entered has duplicate elements")
    else:
        print("The array entered has no duplicate elements")

if __name__ == '__main__':
    main()