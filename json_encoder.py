import json
class solution:
    def myEncoder(self, s):
        if isinstance(s, (list, tuple)):
            temp = []
            for i in s:
                temp.append(self.myEncoder(i))
            return '[' + ', '.join(temp) + ']'
        elif isinstance(s, dict):
            d = '{'
            for i in s:
                print s[i]
                d += self.myEncoder(i) + ': ' + self.myEncoder(s[i])
            d+= '}'
            return d
        elif isinstance(s, bool):
            print 'found bool'
            if s:
                return 'true'
            else:
                return 'false' 
        elif isinstance(s, str):
            return '"%s"'%(s)
        elif isinstance(s, (int, long, float, complex)):
            return str(s)
        elif s is None:
            return 'null'
        else:
            return s
        return res
source = ['foo', {'bar': ('baz', None, 1.0, 2, {'cat': True})}]
#source = ['foo', 'bar']
solution = solution()
output1 = solution.myEncoder(source)
output2 = json.dumps(source)
print 'source:  ', source
print 'output1: ', output1
print 'output2: ', output2
