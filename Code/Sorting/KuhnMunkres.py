import numpy as np
import itertools as it
from copy import deepcopy
np.core.arrayprint._line_width = 120

class KuhnMunkres:
    '''
    Implementation of the Hungarian Algorithm.
    See http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
    '''
    def __init__(self, Matrix):
        self.M   = deepcopy(Matrix)
        self.C   = deepcopy(Matrix)
        self.d   = min(Matrix.shape)
        
        self.Starred = [ [], [] ]
        self.Primed  = [ [], [] ]
        self.cov_Row = list()
        self.cov_Col = list()
        
    def indices(self, List):
        X = List[0]
        Y = List[1]
        Indices = list( (X[i], Y[i]) for i in range(len(X)))
        return Indices

    def vectors(self):
        X = self.d
        Vectors = list( it.product(range(X), repeat=2))
        return Vectors
    
    def if_covered(self, xy):
        if xy[0] not in self.cov_Row and xy[1] not in self.cov_Col:
            return 0
        else:
            return 1

    def star_xy(self, xy):
        self.Starred[0].append( xy[0] )
        self.Starred[1].append( xy[1] )

    def prime_xy(self, xy):
        self.Primed[0].append( xy[0] )
        self.Primed[1].append( xy[1] )
        
    def unstar_xy(self, xy):
        self.Starred[0].remove( xy[0] )
        self.Starred[1].remove( xy[1] )


    def Substract(self):           
        for Line in self.M:
            Line -= min(Line)

    def Star(self):
        Zeros = self.indices( np.where(self.M == 0) )           
        for i in Zeros:
            if i[0] not in self.Starred[0] and i[1] not in self.Starred[1]:
                self.star_xy(i)
                
    def Cover(self):
        for i in self.Starred[1]:
            if i not in self.cov_Col:
                self.cov_Col.append(i)
                
        return len(self.M) - len(self.cov_Col)

    def Prime(self):
        Zeros = self.indices( np.where(self.M == 0) )
        Zeros = list( z for z in Zeros if not self.if_covered(z) )
        while Zeros:
            i = Zeros[0]
            self.prime_xy(i)
            if i[0] in self.Starred[0]:
                self.cov_Row.append( i[0] )
                where = self.Starred[0].index(i[0]) 
                self.cov_Col.remove( self.Starred[1][where] )
                
                Zeros = self.indices( np.where(self.M == 0) )
                Zeros = list( z for z in Zeros if not self.if_covered(z) )
            else:
                return 0
                
        return 1

    def Add(self):
        Min = min( list( self.M[vec] for vec in self.vectors() if not self.if_covered(vec)))
        for i in range(self.d):
            if i in self.cov_Row:
                self.M[i] += Min
            if i not in self.cov_Col:
                self.M[:,i] -= Min

    def Unstar(self):
        oldPrime = (self.Primed[0][-1], self.Primed[1][-1])
        
        while oldPrime[1] in self.Starred[1]:
            self.star_xy( oldPrime)
            
            where   = self.Starred[1].index( oldPrime[1] )
            oldStar = (self.Starred[0][where], oldPrime[1])
            self.unstar_xy( oldStar  )

            where   =   self.Primed[0].index( oldStar[0] )
            oldPrime = ( oldStar[0], self.Primed[1][where])

        self.star_xy( oldPrime )

        self.Primed  = [ [], [] ]
        self.cov_Row = list()
        self.cov_Col = list() 
        
    def run(self):
        self.Substract()
        self.Star()

        while self.Cover():
            while self.Prime():
                self.Add()
            self.Unstar()
            
        Solution = self.indices( self.Starred )
        Solution.sort()

        if len(Solution) != self.d:
            print("Error")
            input()
            
        return Solution

def cost(C, Solution):
    c = 0
    for i in Solution:
        c += C[i]
    return c

  
############################# additional functions that solve KPartite problems #######################
 
def Solve(D):
    '''
    Returns a tupel (cost, Solution)
    '''
    km = KuhnMunkres(D)
    L  = km.run()
    c  = cost(D, L)
    return (c, L)

