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

    """
    Determine whether an integer is a palindrome. Do this without extra space.
    """
    def palindromeNumber(self, num):
	# All negative number consider False
	if num < 0:
	    return False
	# Don't neet to consider overflow
	origin = num
	new = num % 10
	num = num / 10
	while num != 0:
	    new = new * 10 + num % 10
	    num /= 10
	return new == origin
		
    """
    '.' Matches any single character.
    '*' Matches zero or more of the preceding element.

    The matching should cover the entire input string (not partial).

    The function prototype should be:
        bool isMatch(const char *s, const char *p)

        Some examples:
            isMatch("aa","a") false
            isMatch("aa","aa") true
            isMatch("aaa","aa") false
            isMatch("aa", "a*") true
            isMatch("aa", ".*") true
            isMatch("ab", ".*") true
            isMatch("aab", "c*a*b") true
    """
    def isMatch(self, s, p):
        #use DP, a two-demension array to store the result (e. dp[i+1][j+1] for s[i] and p[j])
        dp = [[False for i in xrange(len(p)+1)] for j in xrange(len(s)+1)] 
        dp[0][0] = True
        for j in range(1, len(p)+1):
            if p[j-1] == '*':
                if j > 1:
                    dp[0][j] = dp[0][j-2]
        for i in range(1, len(s)+1):
            for j in range(1, len(p)+1):
                if p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    dp[i][j] = dp[i][j-1] or dp[i][j-2] or (dp[i-1][j] and (s[i-1] == p[j-2] or p[j-2] == '.'))
                else:
                    dp[i][j] = dp[i-1][j-1] and s[i-1] == p[j-1]
        return dp[len(s)][len(p)]

    """
    Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

    Note: You may not slant the container.
    """
    def maxArea(self, height):
        if len(height) < 2:
            return 0
        l = 0
        r = len(height) - 1
        res = 0
        while l < r:
            area = min(height[l], height[r])*(r-l)
            res = area if area > res else res
            if height[l] > height[r]:
                r -= 1
            else:
                l += 1
        return res
    
    """
    Given an integer, convert it to a roman numeral.

    Input is guaranteed to be within the range from 1 to 3999.
    """
    def intToRoman(self, num):
        if num < 1:
            return None
        values = [ 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 ]
        numerals = [ "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" ]
        res = ''
        for i in xrange(len(values)):
            while num >= values[i]:
                num -= values[i]
                res += numerals[i]
        return res

    """
    Given a roman numeral, convert it to an integer.

    Input is guaranteed to be within the range from 1 to 3999.
    """
    def romanToInt(self, s):
        d = {"M": 1000,  "D": 500, "C": 100, "L": 50, "X": 10, "V": 5 , "I":1}
        res = 0
        pre = 0
        i = len(s)-1
        while i>=0:
            if s[i] in d:
                if d[s[i]] < pre:
                    res -= d[s[i]]
                else:
                    res += d[s[i]]
                pre = d[s[i]]
            else:
                return 0
            i -= 1
        return res
    """
    Write a function to find the longest common prefix string amongst an array of strings.
    """
    def longestCommonPrefix(self, strs):
        if len(strs) == 0:
            return ''
        res = list(strs[0])
        for i in xrange(len(strs)):
            curr = []
            for j in xrange(min(len(res), len(strs[i]))):
                if strs[i][j] == res[j]:
                    curr.append(strs[i][j])
                else:
                    break
            res = curr
        return ''.join(res)

    """
    Given an array S of n integers, are there elements a, b, c in S such that 
    a + b + c = 0? Find all unique triplets in the array which gives the sum of zero. 
    """
    def threeSum(self, nums):
        n = len(nums)
        if n < 3:
            return []
        nums.sort()
        res = []
        for i in xrange(len(nums)-1):
            left = i + 1
            right = len(nums)-1
            while(left<right):
                val = nums[i] + nums[left] + nums[right]
                if val == 0 and [nums[i], nums[left], nums[right]] not in res:
                    res.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -=1
                elif (val<0):
                    left += 1
                else:
                    right -= 1
        return res
            
    """
    Given an array S of n integers, find three integers in S such that 
    the sum is closest to a given number, target. 
    Return the sum of the three integers. 
    You may assume that each input would have exactly one solution.
    """
    def threeSumClosest(self, nums, target):
        n = len(nums)
        if n < 3:
            return None
        nums.sort()
        mindiff = abs(nums[0]+nums[1]+nums[2] - target)
        res = nums[0]+nums[1]+nums[2]
        for i in xrange(n-1):
            left = i + 1
            right = n - 1
            while left < right:
                s = nums[i]+nums[left]+nums[right]
                diff = abs(s-target)
                if diff == 0:
                    return s
                elif s < target:
                    left += 1
                else:
                    right -= 1
                if diff < mindiff:
                    mindiff = diff
                    res = s
        return res

    """
    Given a digit string, return all possible letter combinations that the number could represent.

    A mapping of digit to letters (just like on the telephone buttons) is given below.
    """
    def letterCombination(self, digits):
	#digits = list(digits)
	def dfs(index, string, res):
            if index == len(digits):
		res.append(string)
		print res
		return
	    if digits[index] in digitMap:
	        for i in digitMap[digits[index]]:
		    #string.append(i)
		    dfs(index+1, string+i, res)
	    else:
		return ''	    
    	digitMap = {
	    '2':['a','b','c'],
            '3':['d','e','f'],
            '4':['g','h','i'],
            '5':['j','k','l'],
            '6':['m','n','o'],
            '7':['p','q','r','s'],
            '8':['t','u','v'],
            '9':['w','x','y','z']
        }
	res = []
	dfs(0, '', res)
	return res    	    

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

    num = 123454321
    result9 = s.palindromeNumber(num)
    s.p('Palindrome Number', num, str(result9))

    ori = 'aab'
    sec = 'c*a*b'
    result10 = s.isMatch(ori, sec)
    s.p('Regular Expression Match', (ori,';', sec), 'Match' if result10 else 'Dont Match')

    height = [3,7,2,7,9,5,5]
    result11 = s.maxArea(height)
    s.p('Container with most water', height, result11)
    
    num = 3591
    result12 = s.intToRoman(num)
    s.p('Int to Roman', num, result12)

    string = 'XCIX'
    result13 = s.romanToInt(string)
    s.p('Roman To Int', string, result13)

    strs = ['aca', 'acba']
    result14 = s.longestCommonPrefix(strs)
    s.p('Longest Common Prefix', strs, result14)

    nums = [-1, 0, 1, 2, -1, -4]
    result15 = s.threeSum(nums)
    s.p('Three Sum', nums, result15)

    nums = [-1, 2, 1, -4]
    result16 = s.threeSumClosest(nums, 1)
    s.p('Three Sum Closest', nums, result16)

    digits = '23'
    result17 = s.letterCombination(digits)
    s.p('Letter Combination', digits, result17)
