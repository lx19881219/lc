import sys
 
class lnode:

    def __init__(self, value=None):
	self.val = value
	self.nxt = None

    def setNext(self, node):
	self.nxt = node

    def print_list(self, head):
	if head != None:
	    print head.val,
	    return self.print_list(head.nxt)
	print ''

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

    """
    You are given two linked lists representing two non-negative numbers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

    Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    Output: 7 -> 0 -> 8

    """
    def addTwoNums(self, num1, num2):
	head = lnode()
	flag = 0
	cur = head
	while num1 and num2:
	    cur.nxt = lnode((num1.val + num2.val + flag) % 10)
	    flag = (num1.val + num2.val + flag) / 10
	    num1 = num1.nxt
	    num2 = num2.nxt
	    cur = cur.nxt
	while num1:
	    cur.nxt = lnode((num1.val + flag) % 10)
	    flag = (num1.val + flag) / 10
	    num1 = num1.nxt
	    cur = cur.nxt
	while num2:
	    cur.nxt = lnode((num2.val + flag) % 10)
	    flag = (num2.val + flag) / 10
	    num2 = num2.nxt
	    cur = cur.nxt    
	if flag == 1:
	    cur.nxt = lnode(1)
	return head.nxt

    """
    Given a string, find the length of the longest substring without repeating characters. For example, the longest substring without repeating letters for "abcabcbb" is "abc", which the length is 3. For "bbbbb" the longest substring is "b", with the length of 1.
    """
    def longest_substring(self, string):
	d = {}
	res = 0
	for i in xrange(len(string)):
	    if string[i] in d:
		distance = i - d[string[i]]
		if distance > res:
		    res = distance
	    d[string[i]] = i
	return res

    """
    There are two sorted arrays nums1 and nums2 of size m and n respectively. Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
    """
    def getMedian(self, nums1, nums2, k):
	l1 = len(nums1)
	l2 = len(nums2)
	if l1 > l2:
	    return self.getMedian(nums2, nums1, k)
	if l1 == 0:
	    return nums2[k-1]
	if k == 1:
	    return min(nums1[0], nums2[1])
	p1 = min(k/2, l1)
	p2 = k - p1
	if nums1[p1-1] <= nums2[p2-1]:
	    return self.getMedian(nums1[p1:], nums2, k-p1)
	else:
	    return self.getMedian(nums1, nums2[p2:], k-p2)
	    	

    def medianOfSorted(self, nums1, nums2):
	m = len(nums1)
	n = len(nums2)
	if (m+n) % 2 == 1:
	    return (self.getMedian(nums1, nums2, (m+n)/2) + self.getMedian(nums1, nums2, (m+n)/2+1))/2
	else:
	    return self.getMedian(nums1, nums2, (m+n)/2)

    """
    Given a string S, find the longest palindromic substring in S. You may assume that the maximum length of S is 1000, and there exists one unique longest palindromic substring.
    """
    def getPalindromic(self, string, l, r):
	while l >= 0 and r < len(string) and string[l] == string[r]:
	    l -= 1
	    r += 1
	return string[l+1:r]

    def longestPalindromic(self, string):
	start = 0
	end = 0
	res = ''
	for i in xrange(len(string)):
	    if i<len(string)-1 and string[i] == string[i+1]:
		sub = self.getPalindromic(string, i, i+1)
	    else:
		sub = self.getPalindromic(string, i, i)
	    if len(sub) > len(res):
		res = sub
	return res

    """
     The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

    P   A   H   N
    A P L S I I G
    Y   I   R

    And then read line by line: "PAHNAPLSIIGYIR"

    Write the code that will take a string and make this conversion given a number of rows:

    string convert(string text, int nRows);

    convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR". 
    """
    def zigzag(self, string, nRows):
	return 'Confused'

    """
    Reverse digits of an integer.

    Example1: x = 123, return 321
    Example2: x = -123, return -321
    """
    def reverseInt(self, n):
	if n < 0:
	    flag = -1
	else:
	    flag = 1
	x = abs(n)
	res = 0
	while x>0:
	    res = res*10 + x%10
	    x/=10
	if res > 2**31-1: #Consider overflow in other languages, Integer.MAX_VALUE in java
	    return 0
	else:
	    return flag*res

    """
    Implement atoi to convert a string to an integer.
 
    Hint: Carefully consider all possible input cases. If you want a challenge, please do not see below and ask yourself what are the possible input cases.

    Notes: It is intended for this problem to be specified vaguely (ie, no given input specs). You are responsible to gather all the input requirements up front.    
    """
    def atoi(self, string):
	l = len(string)
	INT_MAX = 2**31-1
	sign = 1
	flag = True
	res = 0
	for i in xrange(l):
	    if string[i].isspace():
		continue
	    elif string[i] == '+' and flag:
		flag = False
	    elif string[i] == '-' and flag:
		sign = -1
		flag = False
	    elif string[i].isdigit():
		res = res*10 + int(string[i])%10
	        if res > INT_MAX:
		    return 0
	    else:
		break
	return sign*res	        
		
    def p(self, function, src, res):
	print '{0}\nInput: {1}\nOutput {2}\n'.format(function, src, res)

if __name__ == "__main__":
    s = solution()
    arr = [2, 7, 11, 15]
    result1 = s.two_sum(arr, 9)
    s.p('Two Sum', arr, result1)

    ln1 = lnode(2)
    ln2 = lnode(4)
    ln3 = lnode(3)
    ln1.setNext(ln2)
    ln2.setNext(ln3)
    lnode().print_list(ln1)
    ln4 = lnode(5)
    ln5 = lnode(6)
    ln6 = lnode(4)
    ln4.setNext(ln5)
    ln5.setNext(ln6)
    lnode().print_list(ln4)
    result2 = s.addTwoNums(ln1, ln4)
    lnode().print_list(result2)
    
    string = 'abcabcbb'
    #string = ''
    result3 = s.longest_substring(string)
    s.p('Longest Substring', string, result3)

    nums1 = [1,3,7]
    nums2 = [2,4,6,8]
    result4 = s.medianOfSorted(nums1, nums2)
    s.p('Median of two sorted array', nums1+nums2, result4)

    string = 'abaabbaabcbaabcdcba'
    result5 = s.longestPalindromic(string)
    s.p('Longest Palindromic Substring', string, result5)

    string = 'paypalishiring'
    n=3
    result6 = s.zigzag(string, n)
    s.p('Zigzag', string, result6)

    n = 1534236469
    result7 = s.reverseInt(n)
    s.p('Reverse Integer', n, result7)

    string = '    -126489'
    result8 = s.atoi(string)
    s.p('atoi', string, result8)
