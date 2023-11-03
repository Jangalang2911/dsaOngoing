# You are climbing a staircase. It takes n steps to reach the top.

# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

# Example 1:

# Input: n = 2
# Output: 2
# Explanation: There are two ways to climb to the top.
# 1. 1 step + 1 step
# 2. 2 steps
class Solution:

    def climbStairs(self, n: int) -> int:
        # stored_steps = [-1]*(n+1)
        # stored_steps[0] = 1
        # stored_steps[1] = 1
        # for i in range(2, n+1):
        #     stored_steps[i] = stored_steps[i-1]+stored_steps[i-2]
        # return stored_steps[n]

    #uses O(n) space; can be optimized to O(1) space since we only need to 
    #remember the previous two elements. Also optimizes time hugely since 
    #no need to access and store elements in list


        if n==1: return 1

        one_before = 1
        two_before = 1
        for i in range(2, n+1):
            total = one_before+two_before
            two_before = one_before
            one_before = total
        return total
