class solution:
    """
    Given an array of integers, find two numbers such that they add up to a specific target number.

    The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

    You may assume that each input would have exactly one solution.

    Input: numbers={2, 7, 11, 15}, target=9
    Output: index1=1, index2=2
    """
    def two_sum(self, arr, target):
	if len(arr) < 2:
	    return None
	d = {}
	for i in xrange(len(arr)):
	    if target - arr[i] in d.keys():
		return (d[target - arr[i]] + 1, i + 1)
	    else:
	    	d[arr[i]] = i
	return None

    def p(self, function, src, res):
	print '{0}\nInput: {1}\nOutput {2}\n'.format(function, src, res)

if __name__ == "__main__":
    s = solution()
    arr = [2, 7, 11, 15]
    result1 = s.two_sum(arr, 9)
    s.p('Two Sum', arr, result1)
