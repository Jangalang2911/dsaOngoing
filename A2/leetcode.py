class Solution(object):
    #Arrays No.38
    ############
    #More Optimal Solutions:
    # 1. Use regex
    # 2. Itertools.groups

    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n==1:
            return '1'

        else:
            init = self.countAndSay(n-1)
            #process init
            return self.recursive_process(init)
            
    def recursive_process(self, s): 
        homogeneous = True if len(set(s))==1 else False
        ret_val=''
        if homogeneous:
            ret_val+= str(len(s))+s[0] 

        elif not homogeneous:
            delimiters = []
        
            for curr in range(len(s)-1):
                nex = curr+1
                if s[nex]!=s[curr]:
                    delimiters.append(nex)

            for i in range(len(delimiters)):
                if i == 0:
                    ret_val+=self.recursive_process(s[0:delimiters[0]])
                if i==len(delimiters)-1:
                    ret_val+=self.recursive_process(s[delimiters[0]:])
                else:
                    ret_val+=self.recursive_process(s[delimiters[i]:delimiters[i+1]])

        return ret_val