class tree:
    def __init__(self, x):
	self.val = x
	self.left = None
	self.right = None	

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
    
    def print_result(self, function, src, res):
        print '{0}\nInput: {1}\nOutput {2}\n'.format(function, src, res)
    
if __name__ == "__main__":
    arr = [1,2,3,4,2,1,3]
    result1 = solution().single_number_dict(arr)
    solution().print_result('Single Number', arr, result1)

