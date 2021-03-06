import sys
import pdb
import heapq
 
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

class interval:
    def __init__(self, s, e):
        self.start = s
        self.end = e

class solution:
    """
    Given an array of integers, find two numbers such that they add up to a specific target number.

    The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

    You may assume that each input would have exactly one solution.

    Input: numbers={2, 7, 11, 15}, target=9
    Output: index1=1, index2=2
    """
    def two_sum(self, arr, target):
        # Use Dict
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
    def getkth(self, nums1, nums2, k):
	l1 = len(nums1)
	l2 = len(nums2)
	if l1 > l2:
	    return self.getkth(nums2, nums1, k)
        # if nums2 is empty, get kth directly
	if l1 == 0:
	    return nums2[k-1]
	if k == 1:
	    return min(nums1[0], nums2[0])
        # compare num1[k/2] and nums[k/2], remove the smaller one
        # nums1={1,3,5,7};nums2 = {2,4,6,8,9,10} k=7 k/2=3
        # 5 < 6 so remove 1, 3, 5, because it is impossible that the kth in here
	p1 = min(k/2, l1)
	p2 = k - p1
	if nums1[p1-1] <= nums2[p2-1]:
	    return self.getkth(nums1[p1:], nums2, k-p1)
	else:
	    return self.getkth(nums1, nums2[p2:], k-p2)
	    	

    def medianOfSorted(self, nums1, nums2):
	m = len(nums1)
	n = len(nums2)
	if (m+n) % 2 == 1:
	    return (self.getkth(nums1, nums2, (m+n)/2) + self.getkth(nums1, nums2, (m+n)/2+1))/2
	else:
	    return self.getkth(nums1, nums2, (m+n)/2)

    """
    Given a string S, find the longest palindromic substring in S. You may assume that the maximum length of S is 1000, and there exists one unique longest palindromic substring.
    """
    def getPalindromic(self, string, l, r):
	while l >= 0 and r < len(string) and string[l] == string[r]:
	    l -= 1
	    r += 1
	return string[l+1:r]

    def longestPalindromic(self, string):
	res = ''
        for i in range(len(string)):
            sub1 = self.getPalindromic(string, i, i)
            if len(sub1) > len(res):
                res = sub1
            sub2 = self.getPalindromic(string, i, i+1)
            if len(sub2) > len(res):
                res = sub2
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

    """
    Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

    Note:
    """
    def fourSum(self, nums, target):
	nums.sort()
	twoSum = {}
	res = set()
	length = len(nums)
	if length < 4:
	    return None
	# using a dict to store sum of every two element
	for i in xrange(length):
	    for j in xrange(i+1, length):
		s = nums[i] + nums[j]
		if s in twoSum:
		    twoSum[s].append((i, j))
		else:
	       	    twoSum[s] = [(i, j)]
	for i in xrange(length):
	    for j in xrange(i+1, length):
		remain = target - nums[i] - nums[j]
		if remain in twoSum:
		    for keys in twoSum[remain]:
			if keys[0] > j:
			    res.add((nums[i], nums[j], nums[keys[0]], nums[keys[1]]))
		           
	return [list(i) for i in res] 

    """
    Given a linked list, remove the nth node from the end of list and return its head.
    """
    def removeNthFromEnd(self, head, n):
	res = lnode(0)
	res.setNext(head)
	first = res
	second = res
	count = 0
	while first.nxt:
	    if count >= n:
		second = second.nxt
	    first = first.nxt
	    count += 1
	second.nxt = second.nxt.nxt
	return res.nxt

    """
    Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

    The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
    """
    def validPareneses(self, s):
	stack = []
	for i in xrange(len(s)):
	    if s[i] == '(' or s[i] == '[' or s[i] == '{':
		stack.append(s[i])
	    if s[i] == ')':
		if len(stack) == 0 or stack.pop() != '(':
		    return False
	    if s[i] == ']':
		if len(stack) == 0 or stack.pop() != '[':
		    return False
	    if s[i] == '}':
		if len(stack) == 0 or stack.pop() != '{':
		    return False
	if len(stack) != 0:
	    return False
	return True

    """
    Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
    """
    def mergeTwoList(self, l1, l2):
	res = temp = lnode(0)
	while l1 and l2:
	    if l1.val > l2.val:
		temp.nxt = l2
		temp = temp.nxt
		l2 = l2.nxt
	    else:
		temp.nxt = l1
		temp = temp.nxt
		l1 = l1.nxt
	while l1:
	    temp.nxt = l1
	    temp = temp.nxt
	    l1 = l1.nxt
	while l2:
	    temp.nxt = l2
	    temp = temp.nxt
	    l2 = l2.nxt
	return res.nxt
    """
    Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

    For example, given n = 3, a solution set is:

    "((()))", "(()())", "(())()", "()(())", "()()()"
    """
    def dfsForParentheses(self, string, l, r, res):
        if l > r:
            return
        if l == 0 and r == 0:
            res.append(string)
        if l > 0:
            self.dfsForParentheses(string + '(', l-1, r, res)
        if r > 0:
            self.dfsForParentheses(string + ')', l, r-1, res)

    def generateParentheses(self, n):
        res = []
        self.dfsForParentheses('', n, n, res)
        return res
    """
    Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
    """
    def mergeKLists(self, lists):
        heap = []
        head = lnode(0)
        temp = head
        for l in lists:
            if l != None:
                heap.append((l.val, l))
        heapq.heapify(heap)
        while heap:
            pop = heapq.heappop(heap)
            temp.nxt = lnode(pop[0])
            temp = temp.nxt
            if pop[1].nxt != None:
                heapq.heappush(heap, (pop[1].nxt, pop[1].nxt.val))
        return head.nxt

    """
     Given a linked list, swap every two adjacent nodes and return its head.

    For example,
    Given 1->2->3->4, you should return the list as 2->1->4->3.

    Your algorithm should use only constant space. You may not modify the values in the list, only nodes itself can be changed. 
    """
    def swapPairs(self, head):
	if not head.nxt or not head.nxt.nxt:
	    return head
	nHead = lnode(0)
	nHead.nxt = head
	curr = nHead
	
	while curr.nxt and curr.nxt.nxt:
	    tmp = curr.nxt.nxt
	    curr.nxt.nxt = tmp.nxt
	    tmp.nxt = curr.nxt
	    curr.nxt = tmp
	    curr = curr.nxt.nxt

	return nHead.nxt
	
    """
    Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

    If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.

    You may not alter the values in the nodes, only nodes itself may be changed.

    Only constant memory is allowed.
    """
    def reverseKGroup(self, head, k):
	print 'hard 025'

    """
    Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.

    Do not allocate extra space for another array, you must do this in place with constant memory.

    For example,
    Given input array nums = [1,1,2],

    Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the new length. 
    """
    def removeDuplicates(self, nums):
	j = 0
	for i in xrange(0, len(nums)):
	    if nums[i] != nums[j]:
		nums[i], nums[j+1] = nums[j+1], nums[i]
		j += 1
	return j+1
	        
    """
    Given an array and a value, remove all instances of that value in place and return the new length.

    The order of elements can be changed. It doesn't matter what you leave beyond the new length. 
    """
    def removeElement(self, nums, val): # from the begening
	res = 0
	l = 0
	r = len(nums)-1
	while l <= r:
	    if nums[l] == val and nums[r] != val:
		nums[l], nums[r] = nums[r], nums[l]
		l += 1
		r -= 1
	    elif nums[r] == val:
		r -= 1
	    else:
		l += 1
	print nums
	return l

    def removeElementEnd(self, A, elem): # From the end
        j = len(A)-1
        for i in range(len(A) - 1, -1, -1):
            if A[i] == elem:
                A[i], A[j] = A[j], A[i]
                j -= 1
	    print A
        return j+1
    """
    Implement strStr().

    Returns the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack. 

    Brutal match......
    """
    def strStr(self, haystack, needle):
        if len(needle) > len(haystack):
            return -1
	i = 0
        while i < len(haystack) - len(needle) + 1:
            j = 0
            k = i
            while j < len(needle):
                if  haystack[k] == needle[j]: 
                    k += 1
                    j += 1
                else:
                    break
            if j == len(needle):
                break
            else:
                i += 1
        print i
        if i == len(haystack) - len(needle) + 1:
            return -1
        else:
            return i 
    
    """
    Divide two integers without using multiplication, division and mod operator.

    If it is overflow, return MAX_INT. 
    """
    def divide(self, divided, divisor):
        if divisor == 0:
            return 0
        if (divided > 0 and divisor < 0) or (divided < 0 and divisor > 0):
            sign = True
        else:
            sign = False
        a = abs(divided)
        b = abs(divisor)
        res = 0
        while b <= a:
            sum = b
            count = 1
            while sum + sum <= a:
                sum += sum
                count += 1
            a -= sum
            res += count
        if sign:
            return 0-res
        else:
            return res
    """
    You are given a string, s, and a list of words, words, that are all of the same length. Find all starting indices of substring(s) in s that is a concatenation of each word in words exactly once and without any intervening characters.

    For example, given:
    s: "barfoothefoobarman"
    words: ["foo", "bar"]

    You should return the indices: [0,9].
    Scan every m*n long string start from each position in S, see if all the strings in L have been appeared only once using Map data structure. If so, store the starting position.
    """
    def findSubstring(self, s, words):
        if not words:
            return None
        word_len = len(words[0])
        string_len = len(words) * len(words[0])
        word_dict = {}
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
        res = []
        for i in xrange(len(s)):
            k = i
            curr = {}
            print s[k]
            while k+word_len - i <= string_len and s[k:k+word_len] in word_dict:
                print s[k:k+word_len]
                if s[k:k+word_len] not in curr:
                    curr[s[k:k+word_len]] = 1
                else:
                    curr[s[k:k+word_len]] += 1
                if curr[s[k:k+word_len]] > word_dict[s[k:k+word_len]]:
                    break
                if k+word_len - i == string_len:
                    res.append(i)
                    break
                k += word_len

        return res
                
    def findSubstring_list(self, s, words):
        """
        Do not use sth in list, it's O(n), but for sth in dict, it's O(1) on average
        """
        if not words:
            return None
        word_len = len(words[0])
        string_len = len(words) * len(words[0])
        res = []
        for i in xrange(len(s)):
            print s[i]
            k = i
            temp = words[:]
            print k, s[k:k+word_len], temp
            while k+word_len - i <= string_len and s[k:k+word_len] in temp:
                print s[k:k+word_len]
                temp.pop(temp.index(s[k:k+word_len]))
                if len(temp) == 0:
                    res.append(i)
                    break
                k += word_len

        return res

    """
     Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

     If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

     The replacement must be in-place, do not allocate extra memory.

     Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.
     1,2,3 - 1,3,2
     3,2,1 - 1,2,3
     1,1,5 - 1,5,1
    """
    def nextPermutation(self, nums):
        if len(nums) <= 1:
            return nums
        temp = -1
        for i in range(len(nums)-2, -1, -1):
            if nums[i] < nums[i+1]:
                temp = i
                break
        if temp == -1:
            nums.reverse()
            return nums
        else:
            for j in range(len(nums)-1, temp, -1):
                if nums[j] > nums[temp]:
                    nums[j],nums[temp] = nums[temp],nums[j]
                    break
        nums[temp+1:len(nums)] = nums[temp+1:len(nums)][::-1]
        return nums
    
    """
    Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

    For "(()", the longest valid parentheses substring is "()", which has length = 2.

    Another example is ")()())", where the longest valid parentheses substring is "()()", which has length = 4. 
    """
    def longestValidParatheses(self, string):
        """Using Stack"""
        res = 0
        stack = []
        last = -1
        for i in xrange(len(string)):
            if string[i] == '(':
                stack.append(i)
            elif not stack:
                last = i
            else:
                index = stack.pop()
                if not stack:
                    length = i - last
                else:
                    length = i - stack[-1]
                if length > res:
                    res = length
        return res
    """
    Suppose a sorted array is rotated at some pivot unknown to you beforehand.

    (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

    You are given a target value to search. If found in the array return its index, otherwise return -1.

    You may assume no duplicate exists in the array.
    """

    def search(self, nums, target):
        if not nums:
            return -1
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left + right)/2
            if target == nums[mid]:
                return mid
            elif nums[left] <= nums[mid]:
                if target >= nums[left] and target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            elif nums[right] > nums[mid]:
                if target > nums[mid] and target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid -1
        return -1

    """
    Given a sorted array of integers, find the starting and ending position of a given target value.

    Your algorithm's runtime complexity must be in the order of O(log n).

    If the target is not found in the array, return [-1, -1].

    For example,
    Given [5, 7, 7, 8, 8, 10] and target value 8,
    return [3, 4]. 
    """

    def searchRange(self, nums, target):
        """Binary search"""
        if not nums:
            return [-1, -1]
        left = 0
        right = len(nums) - 1
        l = -1
        r = -1
        while left <= right:
            mid = (left + right)/2
            if target == nums[mid]:
                l = mid
                r = mid
                while r < right and nums[mid] == nums[r+1]:
                    r += 1
                while l > left and nums[mid] == nums[l-1]:
                    l -= 1
                return [l, r]
            elif target > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
        return [-1, -1]

    """
    Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

    You may assume no duplicates in the array.

    Here are few examples.
    [1,3,5,6], 5 - 2
    [1,3,5,6], 2 - 1
    [1,3,5,6], 7 - 4
    [1,3,5,6], 0 - 0 
    """

    def searchInsert(self, nums, target):
        if not nums:
            return 0
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left+right)/2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid-1
            else:
                left = mid+1
        return left
    """
    Valid Sudoku
    """
    def isValidSudoku(self, board):
        # validate rows
        for i in xrange(9):
            if board[i].count('.') + len(set(board[i])) - 1 != 9: return False
        # validate columns
        for i in xrange(9):
            col = [board[j][i] for j in xrange(9)]
            if col.count('.') + len(set(col)) - 1 != 9: return False
        # validate 3x3 squares
        for i in (0, 3, 6):
            for j in (0, 3, 6):
                square = [board[i + m][j + n] for m in (0, 1, 2) for n in (0, 1, 2)]
                if square.count('.') + len(set(square)) - 1 != 9: return False
        return True

    """
    Sudoku Solver
    """

    def sudokuSolver(self, board):
        def isValid(x, y):
            tmp = board[x][y]
            board[x][y] = 'E'
            for i in xrange(9):
                if board[x][i] == tmp:
                    return False
            for i in xrange(9):
                if board[i][y] == tmp:
                    return False
            for i in range(3):
                for j in range(3):
                    # SMART!!!!
                    if board[(x/3)*3+i][(y/3)*3+j]==tmp: return False
            board[x][y] = tmp
            return True
        def dfs(board):
            for i in xrange(9):
                for j in xrange(9):
                    if board[i][j] == '.':
                        for k in '123456789':
                            board[i][j] = k
                            if isValid(i, j) and dfs(board):
                                return True
                            board[i][j] = '.'
                        return False
            return True
        dfs(board)
    """
    The count-and-say sequence is the sequence of integers beginning as follows:
    1, 11, 21, 1211, 111221, ...

    1 is read off as "one 1" or 11.
    11 is read off as "two 1s" or 21.
    21 is read off as "one 2, then one 1" or 1211.

    Given an integer n, generate the nth sequence.

    Note: The sequence of integers will be represented as a string. 
    """

    def CountAndSay(self, n):
        if n < 1:
            return ''
        string = '1'
        for c in xrange(n-1):
            tmp = ''
            count = 1
            for i in xrange(len(string)-1):
                if string[i] == string[i+1]:
                    count += 1
                else:
                    tmp += str(count)+string[i]
                    count = 1
            tmp += str(count) + string[-1]
            string = tmp
        return string

    """
     Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

     The same repeated number may be chosen from C unlimited number of times.
    """
    def CombinationSum(self, candidates, target):
        def dfs(candidates, target, nums):
            if target == 0:
                res.append([i for i in nums])
                return
            for i in xrange(len(candidates)):
                if candidates[i] <= target:
                    nums.append(candidates[i])
                    dfs(candidates[i::], target-candidates[i], nums)
                    #clean the num list when find an answer
                    nums.pop()
        candidates.sort()
        res = []
        dfs(candidates, target, [])
        return res 

    """
     Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

     Each number in C may only be used once in the combination. 
    """

    def CombinationSum2(self, candidates, target):
        def dfs(candidates, target, nums):
            if target == 0 and nums not in res:
                res.append([i for i in nums])
                return
            for i in xrange(len(candidates)):
                if candidates[i] <= target:
                    nums.append(candidates[i])
                    dfs(candidates[i+1::], target-candidates[i], nums)
                    #clean the num list when find an answer
                    nums.pop()
        candidates.sort()
        res = []
        dfs(candidates, target, [])
        return res 

    """
     Given an unsorted integer array, find the first missing positive integer.

     For example,
     Given [1,2,0] return 3,
     and [3,4,-1,1] return 2.

     Your algorithm should run in O(n) time and uses constant space. 
    """

    def firstMissingPositive(self, nums):
        """swap element make nums[i] = i+1, loop again find the wrong one"""
        length = len(nums)
        for i in xrange(length):
            while nums[i] != i+1:
                if nums[i] > length or nums[i]<=0 or nums[i] == nums[nums[i]-1]:
                    break
                
                #nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i] this does not work, list out of range????????
                t = nums[nums[i]-1]; nums[nums[i]-1] = nums[i]; nums[i] = t
        for i in xrange(length):
            if nums[i] != i+1:
                return i+1
        return length+1

    """
     Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

     For example,
     Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6. 
    """
    def trap(self, height):
        if not height:
            return 0
        max_height = []
        # loop from left find the max from left
        left_max = 0
        for i in xrange(0,len(height)):
            if height[i] > left_max:
                left_max = height[i]
            max_height.append(left_max)
        # loop from right find the max from right, if < left_max, replace it
        right_max = 0
        for i in xrange(len(height)-1, -1, -1):
            if height[i] > right_max:
                right_max = height[i]
            if right_max < max_height[i]:
                max_height[i] = right_max
        res = 0
        # loop again, calculate the difference between max an current
        for i in xrange(len(height)):
            if max_height[i] > height[i]:
                res += (max_height[i] - height[i])
        return res

    """
    Given two numbers represented as strings, return multiplication of the numbers as a string.

    Note: The numbers can be arbitrarily large and are non-negative.
    """

    def multiply(self, num1, num2):
        num1 = num1[::-1]
        num2 = num2[::-1]
        res = 0
        for i in xrange(len(num1)):
            for j in xrange(len(num2)):
                #print int(num1[i])*int(num2[j])*(10**(i+j))
                res += int(num1[i]) * int(num2[j]) * (10**(i+j))
        return str(res)


    """
    Implement wildcard pattern matching with support for '?' and '*'.
    '?' Matches any single character.
    '*' Matches any sequence of characters (including the empty sequence).

    The matching should cover the entire input string (not partial).

    The function prototype should be:
        bool isMatch(const char *s, const char *p)
    """
    '''def isMatch(self, string, p):
        """confuse........."""
        sp = pp = 0
        star_p = -1
        while sp < len(string):
            if pp < len(p) and (string[sp] == p[pp] or p[pp] == '?'):
                sp += 1
                pp += 1
            elif pp < len(string) and p[pp] == '*':
                sp += 1
                pp += 1
                star_p = pp
            elif star_p != -1:
                sp += 1'''
    """
     Given an array of non-negative integers, you are initially positioned at the first index of the array.

     Each element in the array represents your maximum jump length at that position.

     Your goal is to reach the last index in the minimum number of jumps.

     For example:
     Given array A = [2,3,1,1,4]

     The minimum number of jumps to reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps to the last index.) 
    """
    def jump(self, nums):
        '''res = [i for i in xrange(1<<31-1)]
        for i in xrange(1, len(nums)):
            for j in xrange(i):
                if nums[j] >= i-j:
                    res[i] = min(res[i], res[j]+1)
        return res[-1]'''
        #dp cause TLE

        # We use "last" to keep track of the maximum distance that has been reached
        # by using the minimum steps "ret", whereas "curr" is the maximum distance
        # that can be reached by using "ret+1" steps. Thus,curr = max(i+A[i]) where 0 <= i <= last.
        res = 0
        last = 0
        curr = 0
        for i in xrange(len(nums)):
            if i > last:
                last = curr
                res += 1
                curr = max(curr, i + nums[i])
        return res
    
    """
    Given a collection of numbers, return all possible permutations.

    For example,
    [1,2,3] have the following permutations:
    [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], and [3,2,1]. 
    """
    def permute(self, nums):
        if len(nums) == 0:
            return []
        if len(nums) == 1:
            return [nums]
        res = []
        for i in xrange(len(nums)):
            for j in self.permute(nums[:i]+nums[i+1:]):
                res.append([nums[i]] + j)
        return res

    """
    Given a collection of numbers that might contain duplicates, return all possible unique permutations.

    For example,
    [1,1,2] have the following unique permutations:
    [1,1,2], [1,2,1], and [2,1,1]. 
    """
    def permuteUnique(self, nums):
        # This approach will cause TLE, the solution is to sort before enter
        # the loop, just compare the previous one.
        if len(nums) == 0: return [];
        if len(nums) == 1: return [nums];
        res = []
        for i in xrange(len(nums)):
            for j in self.permuteUnique(nums[:i]+nums[i+1:]):
                # 'in' is O(log(n))
                if [nums[i]] + j in res:
                    continue
                res.append([nums[i]] + j)
        return res
    """
    You are given an n x n 2D matrix representing an image.

    Rotate the image by 90 degrees (clockwise).

    Follow up:
    Could you do this in-place?
    """
    def rotate(self, matrix):
     
        for i in xrange(len(matrix)):
            for j in xrange(i+1, len(matrix)):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
      
        for i in xrange(len(matrix)):
            matrix[i].reverse()
       
        return matrix

    """
    Given an array of strings, group anagrams together.

    For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
    Return:

    [
      ["ate", "eat","tea"],
      ["nat","tan"],
      ["bat"]
    ]

    Note:

    For the return value, each inner list's elements must follow the lexicographic order.
    All inputs will be in lower-case.
    """
    def groupAnagrams(self, strs):
        res = []
        d = {}
        strs.sort()
        for s in strs:
            char_tuple = tuple(sorted(s))
            if char_tuple not in d:
                d[char_tuple] = [s]
            else:
                d[char_tuple].append(s)
        for word in d:
            res.append(d[word])
        return res

    """
    Implement pow(x, n).
    """
    def myPow(self, x, n):
        # x^n = x^(n/2)*x^(n/2)*x^(n%2)
        if n == 0:
            return 1.0
        elif n < 0:
            return 1/myPow(self, x, -n)
        elif n%2:
            return self.myPow(x*x, n/2)*x
        else:
            return self.myPow(x*x, n/2)

    """
    N Queens
    """
    def solveNQueens(self, n):
        def check(m, k):
            # check if it is valid to put queen in row m, column n
            for i in xrange(m):
                if board[i] == k or abs(m-i) == abs(board[i]-k):
                    return False
            return True
        def dfs(m, solution):
            if m == n:
                res.append(solution)
                return
            for i in xrange(n):
                if check(m, i):
                    board[m] = i
                    dfs(m+1, solution+['.'*(i) +'Q' + '.'*(n-i-1)])

        board = [-1 for i in xrange(n)]
        res = []
        dfs(0, [])
        return res

    """
    Follow up for N-Queens problem.

    Now, instead outputting board configurations, return the total number of distinct solutions.
    """
    def totalNQueens(self, n):
        # not done
        def check(m, k):
            for i in range(m):
                if board[i] == k or abs(m-i) == abs(board[i]-k):
                    return False
            return True
        def dfs(m):
            if m == n:
                res.append(1)
                return
            for i in range(n):
                if check(m, i):
                    board[m] = i
                    dfs(m+1)
        board = [-1 for i in range(n)]
        # for res, if define res as res = 0, it will cause referenced before assignment.
        # res is globle and immutable, but list is mutable
        res = []
        dfs(0)
        return len(res)

    """
    Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
    """
    def maxSubarray(self, nums):
        if not nums: return 0;
        temp = 0 
        max_sum = nums[0]
        for i in range(len(nums)):
            if temp < 0:
                temp = 0
            temp += nums[i]
            max_sum = max(temp, max_sum)
        return max_sum
    """
    Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
    """
    def spiralOrder(self, matrix):
        if not matrix: return [];
        l = 0
        u = 0
        d = len(matrix)-1
        r = len(matrix[0])-1
        i = 0
        state = 0 #0:left->right, 1:up->down, 2:right->left, 3:down->up
        res = []
        while True:
            if state == 0:
                for x in range(l, r+1):
                    res.append(matrix[u][x])
                u+=1
            if state == 1:
                for x in range(u, d+1):
                    res.append(matrix[x][r])
                r -=1
            if state == 2:
                for x in range(r, l-1, -1):
                    res.append(matrix[d][x])
                d -= 1
            if state == 3:
                for x in range(d, u-1, -1):
                    res.append(matrix[x][l])
                l += 1
            if l>r or u>d:
                return res
            state += 1
            state %= 4
            i += 1

    """
    Given an array of non-negative integers, you are initially positioned at the first index of the array.

    Each element in the array represents your maximum jump length at that position.

    Determine if you are able to reach the last index.

    For example:
        A = [2,3,1,1,4], return true.

        A = [3,2,1,0,4], return false.
    """
    def canJump(self, nums): 
        if not nums: return False;
        max_len = 0
        for i in range(len(nums)):
            if max_len >= len(nums)-1:
                return True
            if max_len < i:
                return False
            curr_longest = i + nums[i]
            max_len = max(max_len, curr_longest)
        return False

    """
    Given a collection of intervals, merge all overlapping intervals.

    For example,
    Given [1,3],[2,6],[8,10],[15,18],
    return [1,6],[8,10],[15,18]. 
    """
    def merge(self, intervals):
        if not intervals:
            return []
        intervals.sort(key = lambda x: x.start)
        res = []
        res.append(intervals[0])
        for i in xrange(1, len(intervals)):
            # if current in the range of res[-1]
            if intervals[i].end <= res[-1].end:
                continue
            # if current.start between res[-1]
            elif intervals[i].start <= res[-1].end:
                res[-1].end = intervals[i].end
            else:
                res.append(intervals[i])
        return res

    """
    Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.

    If the last word does not exist, return 0.

    Note: A word is defined as a character sequence consists of non-space characters only.

    For example,
    Given s = "Hello World",
    return 5. 
    """
    def lengthOfLastWord(self, s):
        count = 0
        found_word = False
        for i in xrange(len(s)-1, -1, -1):
            if s[i] != ' ':
                count += 1
                found_word = True
            if s[i] == ' ' and found_word:
                break
        return count

    """
    Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order. 
    """
    def generateMatrix(self, n):
        matrix = [[0 for i in xrange(n)] for i in xrange(n)]
        state = 0
        up = left = 0
        right = down = n - 1
        i = 1
        while True:
            if state == 0:
                for j in xrange(left, right + 1):
                    matrix[up][j] = i
                    i += 1
                up += 1
                #state = 1
            elif state == 1:
                for j in xrange(up, down + 1):
                    matrix[j][right] = i
                    i += 1
                right -= 1
                #state = 2
            elif state == 2:
                for j in xrange(right, left-1, -1):
                    matrix[down][j] = i
                    i += 1
                down -= 1
                #state = 3
            elif state == 3:
                for j in xrange(down, up-1, -1):
                    matrix[j][left] = i
                    i += 1
                left += 1
                #state = 0
            if left > right or up > down:
                break
            state += 1
            state %= 4
            for k in matrix:
                print k
        return matrix

    """
    The set [1,2,3,...,n] contains a total of n! unique permutations
    By listing and labeling all of the permutations in order,
    We get the following sequence (ie, for n = 3):

        "123"
        "132"
        "213"
        "231"
        "312"
        "321"

    Given n and k, return the kth permutation sequence.
    """
    def getPermutation(self, n, k):
        # decide sequence char by char
        # for nth num from right, there are (n-1)!+1 selections,
        # k/(n-1)! to decide which number range
        res = ''
        k -= 1
        nums = 1
        for i in xrange(1, n): nums *= i
        num = [1,2,3,4,5,6,7,8,9]
        for i in reversed(range(n)):
            curr = num[k/nums]
            res += str(curr)
            num.remove(curr)
            if i != 0:
                k %= nums
                nums /= i
        return res
    """
    Given a list, rotate the list to the right by k places, where k is non-negative.

    For example:
    Given 1->2->3->4->5->NULL and k = 2,
    return 4->5->1->2->3->NULL.
    """
    def rotateRight(self, head, k):
        # Not Done.....................
        if k == 0 or not head:
            return head
        new_head = lnode(0)
        p = last = head
        count = 0
        while last.nxt:
            last = last.nxt
            count += 1
        step = count - (k % count)
        for i in xrange(step):
            p = p.nxt
        last.nxt = head
        new_head = p.nxt
        p.nxt = None
        return new_head

    """
    A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

    The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

    How many possible unique paths are there?
    """
    def uniquePaths(self, m, n):
        # dp[i][j] = dp[i-1][j] + dp[i][j-1]
        # consider special case first
        if m == 1 or n == 1:
            return 1
        grid = [[0 for i in xrange(n)] for j in xrange(m)]
        for i in xrange(n):
            grid[0][i] = 1
        for j in xrange(m):
            grid[j][0] = 1
        for i in xrange(1, m):
            for j in xrange(1, n):
                grid[i][j] = grid[i-1][j] + grid[i][j-1]
        return grid[m-1][n-1]
    
    """
    Follow up for "Unique Paths":

    Now consider if some obstacles are added to the grids. How many unique paths would there be?

    An obstacle and empty space is marked as 1 and 0 respectively in the grid.
    """
    def uniquePathsWithObstacles(self, obstacleGrid):
        if not obstacleGrid:
            return 0
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0 for i in xrange(n)] for j in xrange(m)]
        for i in xrange(n):
            if obstacleGrid[0][i] == 1:
                dp[0][i] = 0
                break
            else:
                dp[0][i] = 1
        for i in xrange(m):
            if obstacleGrid[i][0] == 1:
                dp[i][0] = 0
                break
            else:
                dp[i][0] = 1
        for i in xrange(1, m):
            for j in xrange(1, n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
    """
    Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

    Note: You can only move either down or right at any point in time.
    """
    def minPathSum(self, grid):
        if not grid:
            return 0
        m = len(grid)
        n = len(grid[0])
        min_map = [[0 for i in xrange(n)] for i in xrange(m)]
        min_map[0][0] = grid[0][0]
        for i in xrange(1, m):
            min_map[i][0] = min_map[i-1][0] + grid[i][0]
        for j in xrange(1, n):
            min_map[0][j] = min_map[0][j-1] + grid[0][j]
        for i in xrange(1, m):
            for j in xrange(1, n):
                min_map[i][j] = min(min_map[i-1][j], min_map[i][j-1]) + grid[i][j]
        return min_map[m-1][n-1]

    """
    Validate if a given string is numeric.
    """
    def isNumber(self, string):
        # Not Done
        print 'isNumber'

    """
    Given a non-negative number represented as an array of digits, plus one to the number.

    The digits are stored such that the most significant digit is at the head of the list.
    """

    def plusOne(self, digits):
        flag = False
        for i in xrange(len(digits)-1, -1, -1):
            if digits[i] == 9:
                digits[i] = 0
                flag = True
            else:
                digits[i] += 1
                flag = False
                break
        if flag:
            digits = [1] + digits
        return digits

    """
    Given two binary strings, return their sum (also a binary string).

    For example,
    a = "11"
    b = "1"
    Return "100". 
    """
    def addBinary(self, a, b):
        la = len(a)-1
        lb = len(b)-1
        flag = 0
        res = ''
        while la >= 0 and lb >= 0:
            sum_digit = int(a[la]) + int(b[lb]) + flag
            new_digit = sum_digit % 2
            if sum_digit > 1:
                flag = 1
            else:
                flag = 0
            res = str(new_digit) + res
            la -= 1
            lb -= 1
        while la >= 0:
            sum_digit = int(a[la]) + flag
            new_digit = sum_digit % 2
            if sum_digit > 1:
                flag = 1
            else:
                flag = 0
            res = str(new_digit) + res
            la -=1
        while lb >= 0:
            sum_digit = int(b[lb]) + flag
            new_digit = sum_digit % 2
            if sum_digit > 1:
                flag = 1
            else:
                flag = 0
            res = str(new_digit) + res
            lb -=1
        if flag == 1:
            res = '1' + res
        return res
    """
    Given an array of words and a length L, format the text such that each line has exactly L characters and is fully (left and right) justified.

    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly L characters.

    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

    For the last line of text, it should be left justified and no extra space is inserted between words.

    For example,
    words: ["This", "is", "an", "example", "of", "text", "justification."]
    L: 16.

    Return the formatted lines as:

    [
        "This    is    an",
        "example  of text",
        "justification.  "
    ]

    Note: Each word is guaranteed not to exceed L in length. 
    """
    def fullJustify(self, words, max_width):
        """begin = 0
        last_word = 0
        spaces = 0
        word = 0
        while word < len(words):
            line = ' ' * max_width
            curr = len(word)
            count = 1
            while (curr + len(word)) <= max_width:
                word += 1
                curr += len(word)
                count += 1
            #last_word = word + 1
            if count == 1:
                space_list.append(' '*(max_width - curr))
            else:
                space = (max_width - curr) / (count - 1)
                for c in xrange(count - 1):
                    if (max_width - curr) % (count - 1) == 0:
                        
            space = (max_width - curr) / (count - 1)
            if (max_width - curr) % (count - 1) != 0:
    
            for j in xrange(last_word, word+1):
                line += word[j]
                if count == 1:
                    space = max_width - curr
                    line += ' '*space
                else:
                    space = (max_width - curr) / (count - 1)
                    if (max_width - curr) % (count - 1) == 0:
                        line += ' ' * space
                    else:
                        space += """

    """
    Implement int sqrt(int x).

    Compute and return the square root of x.
    """
    def mySqrt(self, x):
        if x == 0:
            return 0
        left = 1
        # because (x/2+1)**2>x
        right = x/2 + 1
        while left <= right:
            mid = (left+right)/2
            if mid**2 == x:
                return mid
            elif mid**2 > x:
                right = mid - 1
            else:
                left = mid + 1
        return right

    """
    You are climbing a stair case. It takes n steps to reach to the top.

    Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top? 
    """
    def climbStairs(self, n):
        # steps[n] = step[n-1] + step[n-2] + 2
        if n == 0: return 0;
        if n == 1: return 1;
        if n == 2: return 2;
        steps = [0 for i in xrange(n)]
        steps[0] = 1
        steps[1] = 2
        for i in xrange(2, n):
            steps[i] = steps[i-1] + steps[i-2]
        return steps[-1]

    """
    Given an absolute path for a file (Unix-style), simplify it.

    For example,
    path = "/home/", => "/home"
    path = "/a/./b/../../c/", => "/c"
    """
    def simplifyPath(self, path):
        paths = path.split('/')
        stack = []
        for i in xrange(len(paths)):
            if paths[i] == '..':
                if stack:
                    print stack
                    stack.pop()
                else:
                    continue
            elif paths[i] == '.':
                continue
            elif paths[i]:
                stack.append(paths[i])
            else:
                continue
        res = '/' + '/'.join(stack)
        return res
    """
     Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)

     You have the following 3 operations permitted on a word:

         a) Insert a character
         b) Delete a character
         c) Replace a character
    """
    def minDistance(self, word1, word2):
        # DP, function: dp[i][j]  = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if word1[i-1]==word2[j-1] else 1))
        # where dp[i][j] is the distance from word1[0..i-1] to word2[0..j-1]
        return None
    """
    Given a m x n matrix, if an element is 0, set its entire row and column to 0
    Do it in place. 
    """

    def setZeros(self, matrix):
        rownum = len(matrix)
        colnum = len(matrix[0])
        row = [False for i in xrange(rownum)]
        col = [False for i in xrange(colnum)]
        for i in xrange(rownum):
            for j in xrange(colnum):
                if matrix[i][j] == 0:
                    row[i] = True
                    col[j] = True
        for i in xrange(rownum):
            for j in xrange(colnum):
                if i or j:
                    matrix[i][j] = 0
    """
    Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

    Integers in each row are sorted from left to right.
    The first integer of each row is greater than the last integer of the previous row.

    """
    def searchMatrix(self, matrix, target):
        '''m = len(matrix)
        n = len(matrix[0])
        up = left = 0
        down = m - 1
        right = n - 1'''
        i = 0
        j = len(matrix[0]) - 1
        while i < len(matrix) and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            else:
                i += 1
        return False

    """
     Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.

     Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

     Note:
         You are not suppose to use the library's sort function for this problem. 
    """
    def sortColors(self, nums):
        l = 0
        r = len(nums) - 1
        i = 0
        while i < r:
            if nums[i] == 0:
                nums[i], nums[l] =  nums[l], nums[i]
                l += 1
                i += 1
            elif nums[i] == 2:
                nums[i], nums[r] =  nums[r], nums[i]
                r -= 1
            else:
                i += 1
    """
     Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

     For example,
     S = "ADOBECODEBANC"
     T = "ABC"

     Minimum window is "BANC".

     Note:
     If there is no such window in S that covers all characters in T, return the empty string "".

     If there are multiple such windows, you are guaranteed that there will always be only one unique minimum window in S. 
    """
    def minWindow(self, s, t):
        #TODO double pointer???
        d = {}
        for c in t:
            if c not in d:
                d[c] = 1
            else:
                d[c] += 1
        res = []
                

    """
    Given two integers n and k, return all possible combinations of k numbers out of 1 ... n. 
    """
    def combine(self, n, k):
        def dfs(start, l):
            if len(l) == k:
                res.append(l) 
                return
            for i in xrange(start, n+1):
                dfs(i+1,l + [i])
        res = []
        dfs(1, [])
        return res
    """
     Given a set of distinct integers, nums, return all possible subsets.

     Note:

    Elements in a subset must be in non-descending order.
    The solution set must not contain duplicate subsets.

    """
    def subsets(self, nums):
        def dfs(start, l):
            res.append(l)
            if start == len(nums):
                return
            for i in xrange(start, len(nums)):
                dfs(i+1, l+[nums[i]])
        nums.sort()
        res = []
        dfs(0, [])
        return res

    """
     Given a 2D board and a word, find if the word exists in the grid.

     The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. 
     The same letter cell may not be used more than once. 
    """
    def exist(self, board, word):
        def dfs(curr, x, y):
            if curr == len(word):
                return True
            if x > 0 and board[x-1][y] == word[curr]:
                temp = board[x][y]
                board[x][y] = '.'
                if dfs(curr+1, x-1, y): return True
                board[x][y] = temp
            if x < len(board)-1 and board[x+1][y] == word[curr]:
                temp = board[x][y]
                board[x][y] = '.'
                if dfs(curr+1, x+1, y): return True
                board[x][y] = temp
            if y > 0 and board[x][y-1] == word[curr]:
                temp = board[x][y]
                board[x][y] = '.'
                if dfs(curr+1, x, y-1): return True
                board[x][y] = temp
            if y < len(board[0])-1 and board[x][y+1] == word[curr]:
                temp = board[x][y]
                board[x][y] = '.'
                if dfs(curr+1, x, y+1): return True
                board[x][y] = temp
            return False
        for i in xrange(len(board)):
            for j in xrange(len(board[0])):
                if board[i][j] == word[0]:
                    return dfs(1, i, j)
        return False

    """
     Follow up for "Remove Duplicates":
     What if duplicates are allowed at most twice?

     For example,
     Given sorted array nums = [1,1,1,2,2,3],

     Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3. It doesn't matter what you leave beyond the new length. 
    """
    def removeDuplicates(self, nums):
        if len(nums)< 2:
            return len(nums)
        front = 2
        back = 1
        while front < len(nums):
            if not (nums[front] == nums[back] and nums[front] == nums[back-1]):
                back += 1
                nums[back] = nums[front]
            front += 1    
        return back + 1

    """
    Follow up for "Search in Rotated Sorted Array":
    What if duplicates are allowed?

    Would this affect the run-time complexity? How and why?

    Write a function to determine if a given target is in the array.
    """
    def searchRotateII(self, nums, target):
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left+right)/2
            if nums[mid] == target:
                return True
            if nums[left]==nums[mid]==nums[right]:
                left += 1
                right -= 1
            elif nums[mid] >= nums[left]:
                if nums[mid]>target>=nums[left]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[right]>=target>nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
        return False

    """
     Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.

     For example,
     Given 1->2->3->3->4->4->5, return 1->2->5.
     Given 1->1->1->2->3, return 2->3. 
    """
    def deleteDuplicatesList(self, head):
        if not head.nxt:
            return head
        dummy = lnode(0)
        dummy.nxt = head
        prev = dummy
        curr = dummy.nxt
        while prev.nxt:
            while curr.nxt and curr.nxt.val == prev.nxt.val:
                curr = curr.nxt
            if curr == prev.nxt:
                prev = prev.nxt
                curr = prev.nxt
            else:
                prev.nxt = curr.nxt
        return dummy.nxt

    """
     Given a sorted linked list, delete all duplicates such that each element appear only once.

     For example,
     Given 1->1->2, return 1->2.
     Given 1->1->2->3->3, return 1->2->3. 
    """
    def deleteDuplicates(self, head):

        if not head or not head.nxt:
            return head
        p = head
        while p.nxt:
            if p.val == p.nxt.val:
                p.nxt = p.nxt.nxt
            else:
                p = p.nxt
        return head

    """
    Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram. 
    """
    def largestRectangleArea(self, height):
        res = 0
        index = 0
        h = height[0]
        
    """
     Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing all ones and return its area.
    """
    def maximalRectangle(self, matrix):
        print ' 085 later'

    """
    Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

    You should preserve the original relative order of the nodes in each of the two partitions.

    For example,
    Given 1->4->3->2->5->2 and x = 3,
    return 1->2->2->4->3->5. 
    """

    def partition(self, head, x):
        if not head:
            return head
        l1 = lnode(0)
        l2 = lnode(0)
        res = l1
        temp = l2
        while head:
            lnode().print_list(head)
            if head.val < x:
                l1.nxt = head
                head = head.nxt
                l1 = l1.nxt
                l1.nxt = None
            else:
                l2.nxt = head
                head = head.nxt
                l2 = l2.nxt
                l2.nxt = None
            #head = head.nxt
        l1.nxt = temp.nxt
        return res.nxt
    """
    Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1. 
    """
    def isScramble(self, s1, s2):
        if len(s1) != len(s2):
            return False
        if s1 == s2:
            return True
        for i in xrange(1, len(s1)):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:],s2[i:]):
                return True
            if self.isScramble(s1[:i], s2[len(s1)-i:]) and self.isScramble(s1[i:], s2[:len(s1)-i]):
                return True
        return False

    """
    Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

    Note:
    You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.
    """
    def mergeSortedArray(self, nums1, m, nums2, n):
        i, j, k = m-1, n-1, m+n-1
        while i>=0 and j>=0:
            if nums1[i]>nums2[j]:
                nums1[k] = nums1[i]
                i-=1
            else:
                nums1[k] = nums2[j]
                j-=1
            k -= 1
        while j>=0:
            nums1[k] = nums2[j]
            j-=1
            k-=1

    """
    The gray code is a binary numeral system where two successive values differ in only one bit.

    Given a non-negative integer n representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0.

    For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:

        00 - 0
        01 - 1
        11 - 3
        10 - 2

        Note:
        For a given n, a gray code sequence is not uniquely defined.

        For example, [0,2,3,1] is also a valid gray code sequence according to the above definition.

        For now, the judge is able to judge based on one instance of gray code sequence. Sorry about that.
    """
    def grayCode(self, n):
        # Math Magic 
        res = []
        size = 1<<n
        print size
        for i in xrange(size):
            res.append((i>>1)^i)
            print res
        return res

    """
     Given a collection of integers that might contain duplicates, nums, return all possible subsets.

     Note:

        Elements in a subset must be in non-descending order.
        The solution set must not contain duplicate subsets.

    """
    def subsetWithDup(self, nums):
        def dfs(depth, pos, subset):
            if subset not in res:
                res.append(subset)
            if depth == len(nums):
                return
            for i in xrange(pos, len(nums)):
                dfs(depth + 1, i+1, subset + [nums[i]])
        nums.sort()
        res = []
        dfs(0, 0, [])
        return res

    """
    Given a string containing only digits, restore it by returning all possible valid IP address combinations.

    For example:
        Given "25525511135",

        return ["255.255.11.135", "255.255.111.35"]. (Order does not matter) 
    """
    def restoreIpAddress(self, string):
        def dfs(remain, dot_number, ip):
            if dot_number == 4:
                if remain == '':
                    res.append(ip[1:])
                return 
            for i in xrange(1, 4):
                if i <= len(remain) and int(remain[:i]) <= 255:
                    if str(int(remain[:i])) == remain[:i]:
                        dfs(remain[i:], dot_number+1, ip + '.'+ remain[:i])
                    else:
                        break
        res = []
        '''if len(s) < 4 or len(s) > 12:
            return res'''
        dfs(string, 0, '')
        return res

    """
    Given n, generate all structurally unique BST's (binary search trees) that store values 1...n.
    """
    def generateTrees(self, n):
        def dfs(start, end):
            if start > end:
                return [None]
            res = []
            for rootval in xrange(start, end+1):
                left_tree = dfs(start, rootval-1)
                right_tree = dfs(rootval+1, end)
                for i in left_tree:
                    for j in right_tree:
                        root = tnode(rootval)
                        root.left = i
                        root.right = j
                        res.append(root)
            return res
        return dfs(1, n)

    """
    Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
    """
    def numTrees(self, n):
        # DP dp[0] = 0, dp[1] = 1, dp[2] = 1, dp[n]=dp[0]*dp[n-1]+dp[1]*dp[n-2]+...+dp[n-1]*dp[0]
        dp = [1,1,2]
        if n<=2:
            return dp[n]
        dp += [0 for i in xrange(n-2)]
        for i in xrange(3, n+1):
            for j in xrange(1, i+1):
                dp[i] += dp[j-1]*dp[i-j]
        return dp[n]

    """
     Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.

     For example,
     Given:
         s1 = "aabcc",
         s2 = "dbbca",

         When s3 = "aadbbcbcac", return true.
         When s3 = "aadbbbaccc", return false. 
    """
    def isInterleave(self, s1, s2, s3):
        # DP using a 2D array to store result
        if len(s1)+len(s2) != len(s3):
            return False
        dp = [[False for i in xrange(len(s2)+1)] for i in xrange(len(s1)+1)]
        dp[0][0] = True
        for i in xrange(1, len(s1)+1):
            dp[i][0] = dp[i-1][0] and s3[i-1] == s1[i-1]
        for i in xrange(1, len(s2)+1):
            dp[0][i] = dp[0][i-1] and s3[i-1] == s2[i-1]
        for i in xrange(1, len(s1)+1):
            for j in xrange(1, len(s2)+1):
                dp[i][j] = (dp[i-1][j] and s1[i-1]==s3[i+j-1]) or (dp[i][j-1] and s2[j-1]==s3[i+j-1])
        return dp[len(s1)][len(s2)]

    """
    Given a binary tree, determine if it is a valid binary search tree (BST).

    Assume a BST is defined as follows:

    The left subtree of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.

    """
    def isValidBST(self, root):
        def checkBST(root, minimum, maximum):
            if root == None:
                return True
            if root.val <= minimum or root.val >= maximum:
                return False
            return checkBST(root.left, minimum, root.val) and checkBST(root.right, root.val, maximum)
        return checkBST(root, -10**10, 10**10)

    """
     Two elements of a binary search tree (BST) are swapped by mistake.

     Recover the tree without changing its structure.
     Note:
         A solution using O(n) space is pretty straight forward. Could you devise a constant space solution? 
    """
    '''def recoverTree(self, root):
         def checkBST(root, minimum, maximum):
            if root == None:
                return
            if root.val <= minimum:
                n1 = root    
            elif root.val >= maximum:
                n2 = root
                return
            checkBST(root.left, minimum, root.val)
            checkBST(root.right, root.val, maximum)
        n1 = n2 = None
        checkBST(root, -10**10, 10**10)
        if n1 and n2:
            n1.val, n2.val = n2.val, n1.val
        return root'''
    
    """
     Given two binary trees, write a function to check if they are equal or not.

     Two binary trees are considered equal if they are structurally identical and the nodes have the same value. 
    """
    def isSameTree(self, root1, root2):
        if root1 is None and root2 is None:
            return True
        if root1 and root2 and root1.val == root2.val:
            return self.isSameTree(root1.left, root2.left) and self.isSameTree(root1.right, root2.right)
        return False

    """
    Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
    """
    def isSymmetric(self, root):
        def check(p, q):
            if p is None and q is None:
                return True
            if p and q and p.val == q.val:
                return check(p.left, q.right) and check(p.right, q.left)
            return False
        if root:
            return check(root.left, root.right)
        return True

    """
    Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
    """
    def levelOrder(self, root):
        def dfs(root, level):
            if root is None:
                return
            if level < len(res):
                res[level].append(root.val)
            else:
                res.append([root.val])
            dfs(root.left, level + 1)
            dfs(root.right, level + 1)
        res = []
        dfs(root, 0)
        return res

    """
    Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).
    """
    def zigzagLevelOrder(self, root):
        def dfs(root, level):
            if root is None:
                return
            if level < len(res):
                if level % 2 == 0:
                    res[level].append(root.val)
                else:
                    res[level] = [root.val] + res[level]
            else:
                res.append([root.val])
            dfs(root.left, level + 1)
            dfs(root.right, level + 1)
        res = []
        dfs(root, 0)
        return res

    """
    Given a binary tree, find its maximum depth.

    The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
    """
    def maxDepth(self, root):
        if root is None:
            return 0
        else:
            return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

    def maxDepth_iter(self, root):
        # res list record the deepest node
        if root == None:
            return 0
        stack = [root]
        depth = [1]
        max_depth = 0
        res = []
        while stack:
            node = stack.pop()
            d = depth.pop()
            if d > max_depth:
                res = [node]
                max_depth = d
            elif d == max_depth:
                res.append(node)
            if node.left:
                stack.append(node.left)
                depth.append(d+1)
            if node.right:
                stack.append(node.right)
                depth.append(d+1)
        return max_depth
    """
    Given inorder and postorder traversal of a tree, construct the binary tree.

    Note:
        You may assume that duplicates do not exist in the tree. 
    """
    def buildTree(self, inorder, postorder):
        if not inorder:
            return None
        if len(inorder) == 1:
            return ListNode(inorder[0])
        root = ListNode(postorder[-1])
        root_index = inorder.index(postorder[-1])
        root.left = self.buildTree(inorder[:root_index], postorder[:root_index])
        root.right = self. buildTree(inorder[root_index+1:], [root_index:len(postorder)-1])
        return root

    """
    Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).
    """
    def levelOrderBottom(self, root):
        def dfs(root, level):
            if root == None:
                return
            if level >= len(res):
                res.append([root.val])
            else:
                res[level].append(root.val)
            dfs(root.left, level+1)
            dfs(root.right, level+1)
        res = []
        dfs(root, 0)
        return reversed(res)
        
    """
    Given a binary tree, find its minimum depth.

    The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
    """
    def minDepth(self, root):
        if root is None:
            return 0
        if root.left is None and root.right is not None:
            return self.minDepth(root.right)+1
        if root.left is not None and root.right is None:
            return self.minDepth(root.left)+1
        return min(self.minDepth(root.left), self.minDepth(root.right))+1
    """
    Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum. 
    """
    def hasPathSum(self, root, target):
        if root is None:
            return False
        if root.left is None and root.right is None:
            return root.val == target
        return self.hasPathSum(root.left, target-root.val) or self.hasPathSum(root.right, target-root.val)

    """
    Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum. 
    """
    def pathSum(self, root, target):
        def dfs(root, target, nums):
            if root is None:
                return None
            if root.left is None and root.right is None and root.val == target:
                res.append(nums+[root.val])
            dfs(root.left, target-root.val, nums+[root.val])
            dfs(root.right, target-root.val, nums+[root.val])
        res = []
        dfs(root, target, [])
        return res

    """
    Given a binary tree, flatten it to a linked list in-place. 
    """
    def flattern(self, root):
        if root is None:
            return
        self.flatten(root.left)
        self.flatten(root.right)
        temp = root
        if temp.left is None:
            return
        temp = temp.left
        while temp.right:
            temp = temp.right
        temp.right = root.right
        root.right = root.left
        root.left = None

    """
    Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.
    """
    def minTotal(self, triangle):
        # use dp
        if not triangle:
            return 0
        dp = [0 for i in xrange(len(triangle))]
        dp[0] = triangle[0][0]
        for i in xrange(1, len(triangle)):
            # traverse from back, it won't broke the prev row's value
            for j in xrange(len(triangle[i])-1, -1, -1):
                if j == 0:
                    dp[j] = dp[j] + triangle[i][j]
                elif j == len(triangle[i])-1:
                    dp[j] = dp[j-1] + triangle[i][j]
                else:
                    dp[j] = min(dp[j-1], dp[j]) + triangle[i][j]
        return min(dp)

    """
    Say you have an array for which the ith element is the price of a given stock on day i.

    If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
    """
    def maxProfit(self, prices):
        if not prices:
            return 0
        curr_min = prices[0]
        res = 0
        for i in xrange(1, len(prices)):
            if prices[i] - curr_min > res:
                res = prices[i] - curr_min
            if prices[i] < curr_min:
                curr_min = prices[i]
        return res

    """
    Say you have an array for which the ith element is the price of a given stock on day i.

    Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
    """
    def maxProfit2(self, prices):
        if not price:
            return 0
        res = 0
        for i in xrange(1, len(prices)):
            if prices[i] > prices[i-1]:
                res += prices[i] - prices[i-1]
        return res
                
    """
    Say you have an array for which the ith element is the price of a given stock on day i.

    Design an algorithm to find the maximum profit. You may complete at most two transactions.

    Note:
    You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
    """
    def maxProfit3(self, prices):
        #two array: f1[i] is max profit from day1 to dayi
        #           f2[i] is max profit from dayi to last day
        if not prices:
            return 0
        f1 = [0 for i in xrange(len(prices))]
        f2 = [0 for i in xrange(len(prices))]
        curr_min = prices[0]
        for i in xrange(1, len(prices)):
            curr_min = min(curr_min, prices[i])
            f1[i] = max(f1[i-1], prices[i] - curr_min)
        curr_max = prices[len(prices)-1]
        for i in xrange(len(prices)-2, -1, -1):
            curr_max = max(curr_max, prices[i])
            f2[i] = max(f2[i+1], curr_max - prices[i])
        res = 0
        for i in xrange(len(prices)):
            res = max(res, f1[i]+f2[i])
        return res

    """
     There are N gas stations along a circular route, where the amount of gas at station i is gas[i].

     You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.

     Return the starting gas station's index if you can travel around the circuit once, otherwise return -1. 
    """
    def canCompleteCircuit(self, gas, cost):
        # if remained gas + gas that can fill can not reach to next station, mark next station as start
        if sum(gas) < sum(cost):
            return -1
        remain = 0
        res = 0
        for i in xrange(len(gas)):
            remain += gas[i]-cost[i]
            if remain < 0:
                res = i+1
                remain = 0
        return res

    """
    Given an array of integers, every element appears twice except for one. Find that single one.

    Note:
        Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory? 
    """
    def singleNumber(self, nums):
        # using xor
        ans = nums[0]
        for i in xrange(len(nums)):
            ans = ans^ nums[i]
        return ans
            
    """
     Given an array of integers, every element appears three times except for one. Find that single one.

     Note:
     Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

     Subscribe to see which companies asked this question

    """
    def singleNumber2(self, nums):
        return 

    """
     Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

     For example, given
     s = "leetcode",
     dict = ["leet", "code"].

     Return true because "leetcode" can be segmented as "leet code". 
    """
    def wordBreak(self, s, wordDict):
        dp = [False for i in xrange(len(s))+1]
        dp[0] = True
        for i in xrange(1, len(s)+1):
            for j in xrange(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
        return dp[len(s)]

    """
     Given a linked list, determine if it has a cycle in it.

     Follow up:
         Can you solve it without using extra space? 
    """
    def hasCycle(self, head):
        first = head
        second = head
        while first and first.next:
            first = first.next.next
            second = second.next
            if first == second:
                return True
        return False

    """
     Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

     Note: Do not modify the linked list.

     Follow up:
     Can you solve it without using extra space? 
    """
    def detectCycle(self, head):
        first = head
        second = head
        while first and first.next:
            first = first.next.next
            second = second.next
            if first == second:
                return first
        return None

    """
     Given a singly linked list L: L0→L1→…→Ln-1→Ln,
     reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

     You must do this in-place without altering the nodes' values.

     For example,
     Given {1,2,3,4}, reorder it to {1,4,2,3}. 
    """
    def reorderList(self, head):
        stack = []
        fast = head
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        while slow:
            stack.append(slow)
            slow = slow.next
        new = head
        while new:
            if stack:
                end = stack.pop()
                end.next = new.next
                new.next = end
                new = new.next
            elif new.next:
                new.next.next = None
            new = new.next
        return



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

    nums = [1, 0, -1, 0, -2, 2]
    result18 = s.fourSum(nums, 0)
    s.p('Four Sum', nums, result18)
    
    ln1 = lnode(1)
    ln2 = lnode(2)
    ln3 = lnode(3)
    ln4 = lnode(4)
    ln5 = lnode(5)
    ln1.setNext(ln2)
    ln2.setNext(ln3)
    ln3.setNext(ln4)
    ln4.setNext(ln5)
    lnode().print_list(ln1)
    result19 = s.removeNthFromEnd(ln1, 2)
    lnode().print_list(result19)
    
    string = '()[]{}'
    result20 = s.validPareneses(string)
    s.p('valid Pareneses', string, 'Valid' if result20 else 'Invalid')

    ln1 = lnode(2)
    ln2 = lnode(4)
    ln3 = lnode(9)
    ln1.setNext(ln2)
    ln2.setNext(ln3)
    lnode().print_list(ln1)
    ln4 = lnode(5)
    ln5 = lnode(6)
    ln6 = lnode(8)
    ln4.setNext(ln5)
    ln5.setNext(ln6)
    lnode().print_list(ln4)
    result21 = s.mergeTwoList(ln1, ln4)
    lnode().print_list(result21)

    n = 3
    result22 = s.generateParentheses(n)
    s.p('Generate Parentheses', n, result22)

    l = []
    result23 = s.mergeKLists(l)

    ln1 = lnode(1)
    ln2 = lnode(2)
    ln3 = lnode(3)
    ln4 = lnode(4)
    ln5 = lnode(5)
    ln1.setNext(ln2)
    ln2.setNext(ln3)
    ln3.setNext(ln4)
    ln4.setNext(ln5)
    lnode().print_list(ln1)
    result24 = s.swapPairs(ln1)
    lnode().print_list(result24)

    nums = [1,1,2,2,3,4,5,6,6,6,7,8]
    result26 = s.removeDuplicates(nums)
    s.p('Remove Duplicate', nums, result26)

    nums = [1,2,3,1,4,5,1]
    result27 = s.removeElement(nums, 1)
    s.p('Remove Element', nums, result27)

    string = 'mississippi'
    result28 = s.strStr(string, 'issip')
    s.p('StrStr', string, result28)

    divided = 1
    divisor = -1
    result29 = s.divide(divided, divisor)
    s.p('Divide', '1 / -1', result29)

    string = 'barfoothefoobarman'
    words = ['foo', 'bar']
    result30 = s.findSubstring(string, words)
    s.p('findSubstring', string, result30)

    nums = [6,8,7,4,3,2]
    result31 = s.nextPermutation(nums)
    s.p('nextPermutation', nums, result31)
    nums = []

    string = '(()()'
    result32 = s.longestValidParatheses(string)
    s.p('longestValidParatheses', string, result32)
    
    nums = [4,5,6,7,8,0,1,2,3]
    result33 = s.search(nums, 2)
    s.p('search', nums, result33)

    nums = [1,2,3,4,5,5,6,6]
    result34 = s.searchRange(nums, 5)
    s.p('searchRange', nums, result34)

    nums = [1,3,5,6]
    result35 = s.searchInsert(nums, 2)
    s.p('searchInsert', nums, result35)

    n = 2
    result38 = s.CountAndSay(n)
    s.p('CountAndSay', n, result38)
    
    candidates = [2,3,6,7]
    result39 = s.CombinationSum(candidates, 7)
    s.p('CombinationSum', candidates, result39)

    candidates = [10,1,2,7,6,1,5]
    result40 = s.CombinationSum2(candidates, 8)
    s.p('CombinationSum2', candidates, result40)


    nums = [9,4,1,-1,2]
    result41 = s.firstMissingPositive(nums)
    s.p('firstMissingPositive', nums, result41)

    height = [0,1,0,2,1,0,1,3,2,1,2,1]
    result42 = s.trap(height)
    s.p('trap', height, result42)

    num1 = '45'
    num2 = '45'
    result43 = s.multiply(num1,num2)
    s.p('multiply', num1+num2, result43)

    string = 'ababe'
    p = '?b*e'
    #result44 = s.isMatch(string, p)
    s.p('isMatch', string, 'Confusing....')

    nums = [2,3,1,1,4]
    result45 = s.jump(nums)
    s.p('JumpGameII', nums, result45)

    nums = [1,2,3]
    result46 = s.permute(nums)
    s.p('Permutation', nums, result46)

    nums = [1,1,2]
    result47 = s.permuteUnique(nums)
    s.p('PermuteUnique', nums, result47)

    value = 1
    matrix = [[0 for x in xrange(5)] for x in xrange(2)]
    source = matrix
    for i in xrange(2):
        for j in xrange(5):
            matrix[i][j] = value
            value += 1
    s.rotate(matrix)
    s.p('RotateImage', source, matrix)

    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    result49 = s.groupAnagrams(strs)
    s.p('groupAnagrams', strs, result49)

    x = 2
    n = 5
    result50 = s.myPow(x, n)
    s.p('myPow', '%f^%f' % (x, n), result50)

    n = 4
    result52 = s.totalNQueens(n)
    s.p('totalNQueens', n, result52)

    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    result53 = s.maxSubarray(nums)
    s.p('maxsubarray', nums, result53)

    matrix = [
            [1,2,3],
            [4,5,6],
            [7,8,9]
            ]
    result54 = s.spiralOrder(matrix)
    s.p('spiralOrder', matrix, result54)


    nums = [3,2,1,0,4]
    result55 = s.canJump(nums)
    s.p('canJump', nums, result55)

    string = 'hello world'
    result57 = s.lengthOfLastWord(string)
    s.p('length of last word', string, result57)

    n = 4
    result58 = s.generateMatrix(n)
    for i in result58:
        print i

    n = 6
    k = 400
    result60 = s.getPermutation(n, k)
    s.p('getPermutation', [n,k], result60)

    ln1 = lnode(1)
    ln2 = lnode(2)
    ln3 = lnode(3)
    ln4 = lnode(4)
    ln5 = lnode(5)
    ln1.setNext(ln2)
    ln2.setNext(ln3)
    ln3.setNext(ln4)
    ln4.setNext(ln5)
    lnode().print_list(ln1)
    result61 = s.rotateRight(ln1, 2)
    lnode().print_list(result61)
    
    m = 3
    n = 7
    result62 = s.uniquePaths(m, n)
    s.p('uniquePaths', [m, n], result62)

    digits = [9, 9, 9]
    result65 = s.plusOne(digits)
    s.p('plusOne', digits, result65)

    a = '11'
    b = '1'
    result66 = s.addBinary(a, b)
    s.p('addBinary', [a, b], result66)

    x = 9
    result68 = s.mySqrt(x)
    s.p('mySqrt', x, result68)

    n = 3
    result69 = s.climbStairs(n)
    s.p('climbStairs', n, result69)

    path = "/a/./b/../../c/"
    result70 = s.simplifyPath(path)
    s.p('simplifyPath', path, result70)

    matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]]
    target = 20
    result74 = s.searchMatrix(matrix, target)
    s.p('searchMatrix',[matrix, target], result74)

    nums = [0,1,2,2,1,2,2,0,0,0,2,1]
    print nums
    s.sortColors(nums)
    print "sortColors", nums

    n = 4
    k = 2
    result76 = s.combine(n, k)
    s.p('combination', [n,k], result76)

    nums = [1,2,3]
    result77 = s.subsets(nums)
    s.p('subsets', nums, result77)

    nums = [1,1,1,1,2,2,2,3,3,4]
    print nums
    result80 = s.removeDuplicates(nums)
    s.p('removeDuplicates', nums, result80)

    nums = [1,1,2,3,4,5,1,1]
    result81 = s.searchRotateII(nums, 2)
    s.p('searchRotateII', [nums,2], result81)
    
    ln1 = lnode(1)
    ln2 = lnode(2)
    ln3 = lnode(2)
    ln4 = lnode(4)
    ln5 = lnode(5)
    ln1.setNext(ln2)
    ln2.setNext(ln3)
    ln3.setNext(ln4)
    ln4.setNext(ln5)
    lnode().print_list(ln1)
    result82 = s.deleteDuplicatesList(ln1)
    lnode().print_list(result82)
    
    ln1 = lnode(1)
    ln2 = lnode(2)
    ln3 = lnode(2)
    ln4 = lnode(4)
    ln5 = lnode(5)
    ln1.setNext(ln2)
    ln2.setNext(ln3)
    ln3.setNext(ln4)
    ln4.setNext(ln5)
    lnode().print_list(ln1)
    result83 = s.deleteDuplicates(ln1)
    lnode().print_list(result83)
    
    ln1 = lnode(1)
    ln2 = lnode(4)
    ln3 = lnode(3)
    ln4 = lnode(2)
    ln5 = lnode(5)
    ln6 = lnode(2)
    ln1.setNext(ln2)
    ln2.setNext(ln3)
    ln3.setNext(ln4)
    ln4.setNext(ln5)
    ln5.setNext(ln6)
    lnode().print_list(ln1)
    result86 = s.partition(ln1, 3)
    lnode().print_list(result86)

    s1 = "sqksrqzhhmfmlmqvlbnaqcmebbkqfy"
    s2 = "abbkyfqemcqnblvqmlmfmhhzqrskqs"
    result87 = s.isScramble(s1, s2)
    s.p('isScramble', [s1, s2], result87)

    nums1 = [1,3,5,6,7,9,-1,-1,-1,-1]
    nums2 = [2,3,4,5]
    print [nums1, nums2]
    s.mergeSortedArray(nums1, 6, nums2, 4)
    print nums1

    n = 2
    result89 = s.grayCode(n)
    s.p('grayCode', n, result89)

    nums = [1,2,2]
    result90 = s.subsetWithDup(nums)
    s.p('subsetWithDup', nums, result90)

    string = '25525511135'
    result93 = s.restoreIpAddress(string)
    s.p('restureIpAddress', string, result93)

    s1 = "aabcc"
    s2 = "dbbca"
    s3 = "aadbbcbcac"
    result97 = s.isInterleave(s1, s2, s3)
    s.p('isInterleave', [s1, s2, s3], result97)
