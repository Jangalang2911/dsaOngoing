# Given the root of a binary search tree, return a 
# balanced binary search tree with the same node values. 
# If there is more than one answer, return any of them.

# A binary search tree is balanced if the depth of the 
# two subtrees of every node never differs by more than 1.


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

#Greedy algos
class Solution(object):
    def balanceBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        tree_as_list = self.in_order(root)
        return self.build_tree(tree_as_list)

    def in_order(self, root):
        if root==None:
            return []

        left = self.in_order(root.left)
        middle = [root.val]
        right = self.in_order(root.right)
        return left + middle + right

    def build_tree(self, lst):
        if len(lst)==0:
            return None
        if len(lst)==1:
            return TreeNode(lst[0], left=None, right=None)
        
        else:
            mid = len(lst)//2
            left_tree = self.build_tree(lst[:mid])
            right_tree = self.build_tree(lst[mid+1:])
            return TreeNode(lst[mid], left=left_tree, right=right_tree)
        