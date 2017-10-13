import sys
sys.path.append("../lib")

from HMM import HMM
from util import *

def HMMforward(H, O):
    """Return the Forward Matrix M s.t M[t][s] = alpha_t(s)"""
    n = H.numStates()
    m = len(O)
    M = [[0 for y in range(n)] for x in range(m)]
    
    obs = O[0]
    row0(H, M, obs, n)
    
    for i in range(1, m):
        obs = O[i]
        row = i
        rows(H, M, obs, n, row)
    
    if m == 1:
        return M
    return M

def row0(H, M, obs, n):
    for i in range(n):
        prob = H.startProb(i) * H.obsProb(i, obs)
        M[0][i] = prob
        
def rows(H, M, obs, n, row):
    for i in range(n):
        prob = 0
        for j in range(n):
            prob += M[row-1][j] * H.transProb(j, i) * H.obsProb(i, obs)
        
        M[row][i] = prob    



def HMMforwardLog(H, O):
    """Return the Log-Forward Matrix M s.t. M[t][s] = log(alpha_t(s))"""
    n = H.numStates()
    m = len(O)
    M = [[0 for y in range(n)] for x in range(m)]
    
    obs = O[0]
    row02(H, M, obs, n)
    
    for i in range(1, m):
        obs = O[i]
        row = i
        rows2(H, M, obs, n, row)
        
    if m == 1:
        return M
    
    return M

def row02(H, M, obs, n):
    for i in range(n):
        a = log(H.startProb(i))
        b = log(H.obsProb(i, obs))
        #prob = logSum(a, b)
        prob = 0
        prob = a + b
        
        M[0][i] = prob
    
    
        
def rows2(H, M, obs, n, row):
    for i in range(n):
        start = 0
        prob = 0
        L = []
        for j in range(n):
            a = M[row-1][j] 
            b = log(H.transProb(j, i))
            c = log(H.obsProb(i, obs))
            d = a + b + c
            L.append(d)
            
    
        
        start = logSumList(L)
        prob = start
        M[row][i] = prob   
        start = 0
        prob = 0


def test():
    H = HMM("test1.hmm")
    obs = "ACACAC"
    
    M = HMMforward(H,obs)
    MLog = HMMforwardLog(H, obs)  
    
    prettyPrintMatrix(M)
    prettyPrintMatrix(MLog)
    
    return MLog
    
    
