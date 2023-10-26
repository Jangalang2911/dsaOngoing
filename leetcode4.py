# You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].

# Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:

# 0 <= j <= nums[i] and
# i + j < n
# Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].

 

# Example 1:

# Input: nums = [2,3,1,1,4]
# Output: 2
# Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.


#Approach mostly correct, but minor bugs in recursive algorithm
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        graph = {i:[] for i in range(len(nums))}

        for i in range(len(nums)):
            if i == len(nums)-1:break

            num_edges = nums[i]
            for terminal in range(num_edges, 0, -1):
                graph[i].append(terminal)

        return self.shortest_cost_path(graph, 0, len(nums)-1)

    def shortest_cost_path(self, graph, start, end):
        #base case 2
        if graph[start]==[]:
            return None
            
        #base case 1
        if end in graph[start]:
            return 1

        #recursive case
        else:

            curr_min = float('inf')
            for each in graph[start]:
                path = self.shortest_cost_path(graph, each, end)
                if path < curr_min and path is not None: curr_min = path
            return curr_min + 1