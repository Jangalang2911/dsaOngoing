from leetcode1 import *
a = TreeNode(3, None, None)
b = TreeNode(5, None, None)
c = TreeNode(4, a, b)
d = TreeNode(1, None, None)
e = TreeNode(2, d, c)

sol = Solution()
x = sol.balanceBST(e)
print(x.val)
print(x.left.val)
print(x.right.val)