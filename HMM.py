import re
from math import exp
import random

_error_threshold = 0.0001

class HMMException(Exception):
    def __init__(self, msg = ""):
        self.msg = msg
    
    def __str__(self):
        return self.msg

_blank = re.compile("^\s*\n$")
def _next_line(fp):
    """Return the next line of fp that is not either empty or a comment"""
    line = fp.readline()
    while line and ((not line.strip()) or line[0] == '#'):
        line = fp.readline()

    return line.rstrip()

def _randFromCum(V):
    """Pick a random index from a cumulative distribution"""
    r = random.random()
    for i,v in enumerate(V):
        if v > r:
            return i
    return len(V)-1

# Helper functions for checking validity
def _checkVector(V, s):
    """Check that len(V) is of size s, that all elements
    of V are probabilities, and that they sum to 1"""
    assert len(V) == s, "Incorrect length"
    assert all([type(x)==float and 0 <= x <= 1 for x in V]), "Contains non-probabilities"
    assert abs(1 - sum(V)) <= _error_threshold, "Elements do not sum to 1"

class HMM:
    def __init__(self, file):
        """Create HMM from specified file"""
        self.read(file)


    def read(self, file):
        """Read content from file and validate"""
        with open(file) as fp:

            # First read observation names
            line = _next_line(fp)
            self.obs = re.split("\s+", line)
            self.obsD = {x:i for i,x in enumerate(self.obs)}
            self.m = len(self.obs)

            # Next read state names
            line = _next_line(fp)
            self.states = re.split("\s+", line)
            self.statesD = {x:i for i,x in enumerate(self.states)}
            self.n = len(self.states)

            # Next is start-state probabilities
            self.pi = [float(x) for x in re.split("\s+", _next_line(fp))]
            self.pi_cum = [sum(self.pi[:(i+1)]) for i in range(self.n)]
 
            # Next read transition matrix
            self.A = []
            self.A_cum = []
            for i in range(self.n):
                self.A.append([float(x) for x in re.split("\s+", _next_line(fp).rstrip())])
                self.A_cum.append([sum(self.A[-1][:(i+1)]) for i in range(self.n)])

            # Finally: read observation matrix
            self.B = []
            self.B_cum = []
            for i in range(self.n):
                self.B.append([float(x) for x in re.split("\s+", _next_line(fp).rstrip())])
                self.B_cum.append([sum(self.B[-1][:(i+1)]) for i in range(self.m)])

            assert not _next_line(fp), "Extra information in file"
        self.isValid()

    def isValid(self):
        """Checks that the MMM is vald: throws an HMMException if not"""

        if not all([type(x) == str for x in self.obs]):
            raise HMMException("Observations: all observation names must be strings")

        if not all([type(x) == str for x in self.states]):
            raise HMMException("States: all state names must be strings")
        
        try:
            _checkVector(self.pi, self.n)
        except AssertionError as A:
            raise HMMException("pi vector: " + str(A))

        if len(self.A) != self.n:
            raise HMMException("transition matrix: incorrect number of rows")

        for row in self.A:
            try:
                _checkVector(row, self.n)
            except AssertionError as A:
                raise HMMException("transition matrix row: " + str(A))

        if len(self.B) != self.n:
            raise HMMException("observation matrix: incorrect number of rows")

        for row in self.B:
            try:
                _checkVector(row, self.m)
            except AssertionError as A:
                raise HMMException("oservation matrix row: " + str(A))

        return True

    def obs_name(self, i = None):
        """Return the name of observation index i.  Return a list of all observation if i unspecified."""         
        return self.obs if i == None else self.obs[i]

    def state_name(self, i = None):
        """Return the name of state index i.  Return a list of all states if i unspecified.""" 
        return self.states if i == None else self.states[i]

    def transProb(self, s1, s2):
        """Return transition probabiity a_{s1,s2} = P(q_{t+1} = s2 | q_t = s1).
        States may be specified by string name or index."""
        if type(s1) == str:
            assert s1 in self.statesD, "Bad state name: %s" % (s1)
            s1 = self.statesD[s1]
        if type(s2) == str:
            assert s2 in self.statesD, "Bad state name: %s" % (s2)
            s2 = self.statesD[s2]
        assert 0 <= s1 <= self.n and 0 <= s2 <= self.n, "Bad state index"

        return self.A[s1][s2]

    def obsProb(self, s, o):
        """Return observation probability b_s(o) = P(O_t = o | q_t = s)""" 
        if type(s) == str:
            assert s in self.statesD, "Bad state name: %s" % (s)
            s = self.statesD[s]
        if type(o) == str:
            assert o in self.obsD, "Bad state name: %s" % (o)
            o = self.obsD[o]
        assert 0 <= s <= self.n and 0 <= o <= self.m, "Bad state index or observation index"

        return self.B[s][o]

    def startProb(self, s):
        """Return the initial probability pi[s] = P(q_0 = s)"""
        if type(s) == str:
            assert s in self.statesD, "Bad state name: %s" % (s)
            s = self.statesD[s]

        return self.pi[s]

    def generateSeq(self, t, initial_state = None, finish_states = set()):
        """Generate a list of state/obs tupples.
        If the finish_states set is empty, the list will be exactly length t.
        If the finish_states set contains state names, the list will be at least length t -- 
        but must terminate in one of the finish states."""
        assert type(finish_states) == set and finish_states <= set(self.states), "finish_states set must be a subset of the state names"
        assert not initial_state or initial_state in self.states, "initial_state should be a state name"

        initial_state = self.statesD[initial_state] if initial_state else None
        finish_states = set(range(self.n)) if not finish_states else {self.statesD[x] for x in finish_states}
        current_state = initial_state if initial_state else _randFromCum(self.pi_cum)
        R = []
        i = 0
        while i < t or (not current_state in finish_states):
            R.append((current_state, _randFromCum(self.B_cum[current_state])))
            current_state = _randFromCum(self.A_cum[current_state])
            i += 1
        return R

    def numStates(self):
        """Return number os states"""
        return self.n

    def numObs(self):
        """Return number of observations"""
        return self.m


