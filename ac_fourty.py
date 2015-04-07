
"""
Given an array of integers, every element appears twice except for one. Find that single one.

Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory? 
"""

def single_number(arr):
    # using XOR
    # A^A^B^B^C. And that equals (A^A)^(B^B)^C, so makes it 0^0^C, then we get C
    res = 0
    for x in arr:
	res = res ^ x
    return res

def print_result(function, src, res):
    print '{0}\nInput: {1}\nOutput {2}\n'.format(function, src, res)

if __name__ == "__main__":
    arr = [1,2,3,4,2,1,3]
    result1 = single_number(arr)
    print_result('Single Number', arr, result1)
    
