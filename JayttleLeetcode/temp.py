class Solution:
    def subarraySum(self, nums: list[int], k: int) -> int:
        n = len(nums)
        dp = [0] * n
        dp = 1 if nums[0] == k else 0
        for i in range(1, n):
            dp[i] = dp[i-1] + self.subarraySum(nums[:i], k-nums[i])
        return dp[-1]
    
nums = [1,1,1]
k = 2
sol = Solution()
sol.subarraySum(nums, k)