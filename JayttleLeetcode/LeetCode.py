from typing import List,Optional
import collections

class ListNode:
    def _init_(self, val = 0,next = None):
        self.val = val
        self.next = next

class Solution:
    #1.两数之和
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """1.两数之和_暴力求解"""
        n=len(nums)
        for i in range(n):
            for j in range(n):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []
        #时间复杂度：O(N^2) 空间复杂度O(1)
    
    def twoSum_hashtable(self, nums: List[int], target: int) -> List[int]:
        """1.两数之和_哈希表"""
        hashtable = dict()
        for i, num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target-num], i]
            hashtable[nums[i]] = i
        return []
        #时间复杂度：O(N) 空间复杂度O(N)   
    
    #2.两数相加
    def addTwoNumbers(self, l1: Optional[ListNode],l2: Optional[ListNode]) -> Optional[ListNode]:
        """2.两数相加"""
        if not l1:
            return l2
        if not l2:
            return l1
        l1.val += l2.val
        if l1.val >= 10:
            l1.next = self.addTwoNumbers(ListNode(l1.val // 10), l1.next)
            l1.val %= 10
        l1.next = self.addTwoNumbers(l1.next, l2.next)
        return l1
    
    #49.字母异位词
    def groupAnagrams(self, strs:List[str]) -> List[List[str]]:
        """
        49.字母异位词
        import collections and use sorted
        """
        mp = collections.defaultdict(list)

        for st in strs:
            key = "".join(sorted(st))
            mp[key].append(st)
        
        return list(mp.values())