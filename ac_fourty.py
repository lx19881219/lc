import pdb

class tnode:
    def __init__(self, x, left=None, right=None):
	self.val = x
	self.left = left
	self.right = right

    def insert_left(self, x):
	self.left = tnode(x)
	return self.left

    def insert_right(self, x):
	self.right = tnode(x)
	return self.right

class tree:
    def __init__(self):
	self.root = None

    def insert(self, data, left, right):
	self.root = tnode(data, left, right)	

class solution:
    """
    Given an array of integers, every element appears twice except for one. Find that single one.
    
    Note:
    Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
    """
    
    def single_number(self, arr):
        # using XOR
        # A^A^B^B^C. And that equals (A^A)^(B^B)^C, so makes it 0^0^C, then we get C
        res = 0
        for x in arr:
            res = res ^ x
        return res
    
    def single_number_dict(self, arr):
        # Using Dict to store integers, delete when key exists.
        if len(arr) == 0:
            return 0
        d = {}
        for item in arr:
            if item in d.keys():
                d.pop(item, None)
            else:
                d[item] = True
        return d.keys()[0]
    
    """
    Given a binary tree, find its maximum depth.
    
    The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
    """

    def max_depth(self, root):
	#recursively
	'''left = 0
	if root.left:
	    left = self.max_depth(root.left)
	right = 0
	if root.right:
	    right = self.max_depth(root.right)
	return max(left, right) + 1'''
	if root == None:
	    return 0
	return max(self.max_depth(root.left), self.max_depth(root.right)) + 1

    def max_depth_iter(self, root):
	#iteraction
	res = 0
	left = 0
	right = 0	
        stack = []
	stack.append([root, 1])
	while len(stack) > 0:
	    node, cur_depth = stack.pop()
	    res = max(res, cur_depth)
	    if node.left != None:
		stack.append([node.left, cur_depth+1])
	    if node.right != None:
		stack.append([node.right, cur_depth+1])
	return res

    def two_sum_ii(self, arr, target):
	#Greedy Algorithm: search from both end.
	if len(arr) < 2:
	    return None
	i = 0
	j = len(arr) - 1
	while(i < j):
	    if target > arr[i] + arr[j]:
		i += 1
	    elif target < arr[i] + arr[j]:
		j -= 1
	    if target == arr[i] + arr[j]:
		return [i, j]
	return None 
    """
    Given two binary trees, write a function to check if they are equal or not.

    Two binary trees are considered equal if they are structurally identical and the nodes have the same value. 
    """
    def same_tree(self, root1, root2):
	#recursively
	if root1 == None and root2 == None:
	    print 'c1'
	    return True
	'''if root1 == None or root2 == None:
	    print 'c2'
	    return False
	if root1.val != root2.val:
	    return False'''
	if root1 and root2:
	    if root1.val == root2.val:
	        print 'c3', root1.val
	        return  self.same_tree(root1.left, root2.left) and self.same_tree(root1.right, root2.right)
	return False

    """ 
    Say you have an array for which the ith element is the price of a given stock on day i.

    Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
    """

    def max_profit(self, arr):
	if not arr or len(arr) == 1:
	    return 0
	profit = 0
	for i in xrange(1, len(arr)):
	    if arr[i] > arr[i-1]:
		profit += arr[i] - arr[i-1]
	return profit

    """
    Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).

    For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.

    Credits:
    Special thanks to @ts for adding this problem and creating all test cases.
    """
    def num_of_1_bits(self, number):
	

    def print_result(self, function, src, res):
        print '{0}\nInput: {1}\nOutput {2}\n'.format(function, src, res)

    def print_tree(self, root):
	print root.val
	if root.left:
	    print 'left'
	    self.print_tree(root.left)
	if root.right:
	    print 'right'
	    self.print_tree(root.right)
	return True
    
if __name__ == "__main__":

    arr = [1,2,3,4,2,1,3]
    result1 = solution().single_number_dict(arr)
    solution().print_result('Single Number', arr, result1)

    root = tnode(0)
    n1 = root.insert_left(3)
    n2 = root.insert_right(5)
    n3 = n1.insert_left(7)
    n4 = n1.insert_right(9)
    n5 = n4.insert_left(1)
    n6 = n2.insert_left(2)
    """
		0
	    3	    5
	  7   9   2
	     1
	
    """
    solution().print_tree(root)
    result2 = solution().max_depth_iter(root)
    solution().print_result('Max Depth', root.val, result2)

    arr = [1,5,8,12,24]
    result3 = solution().two_sum_ii(arr, 17)
    solution().print_result('Two Sum II', arr, result3)

    root1 = tnode(0)
    n7 = root1.insert_left(3)
    n8 = root1.insert_right(5)
    n9 = n7.insert_left(7)
    n10 = n7.insert_right(9)
    n11 = n10.insert_left(1)
    n12 = n8.insert_left(2)
    solution().print_tree(root1)
    result4 = solution().same_tree(root, root1)
    solution().print_result('Same Tree', 0, result4)

    arr = [2, 5, 1, 8, 2, 1, 4]
    result5 = solution().max_profit(arr)
    solution().print_result('Best Profit', arr, result5)
