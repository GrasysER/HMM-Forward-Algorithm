import math

# Useful constants
inf = float('inf')
neg_inf = -inf

# Useful functions
exp = math.exp


def prettyPrintMatrix(V, width = 8, precision = 4):
    """Pretty-print a matrix"""
    print("\n".join(["".join([ ("{:%d.%df}" % (width, precision)).format(x) for x in row] ) for row in V]))


def log(x):
    """Return log(x), where x >= 0.  (Returns -inf if x == 0.)"""
    return neg_inf if x == 0 else math.log(x)


def logSum(x,y):
    """Compute log(exp(x) + exp(t)) in a numerically stable fasion"""
    if x == neg_inf:
        return y
    if y == neg_inf:
        return x
    return x + log(1 + exp(y-x))

def logSumList(L):
    """Compute log(sum([exp(x) for x in L])) in a numerically stable way"""
    if len(L) == 0:
        raise ValueError("Zero element list")
    if len(L) == 1:
        return L[1]
    z = logSum(L[0], L[1])
    for x in L[2:]:
        z = logSum(z, x)
    return z
