import numpy as np
import itertools as it
import Sorting.KuhnMunkres as mk
import Sorting.H5functions as hf

### return all n-tupels of "from L choose n"
def Pairs(L, n):
    return list(it.combinations(range(L), n))

### called by Total_Cost and Find_Motifs
def difference(M_1,M_2,order=2):
    '''  
    Comparison of two ensembles:
    The ensemble is shifted over the other until the lowest difference occurs
    Hereby the L2-norm is used (order=2) devided by the product of the l0-norms 
    of the ensembles
        
    Parameters
    -----------
    M_1  : Ensemble 1
    
    M_2  : Ensemble 2
    
    order: L-norm of the comparison
    
    Returns
    -----------
    minimum  : the lowest cost of all possible shifts
    '''
    
    (x1,x2) = (len(M_1), len(M_2)) # neurons
    (y1,y2) = (len(M_1[0]), len(M_2[0])) # 3*length
    FillUp  = np.zeros([x2,y2-1])
    N       = np.concatenate([FillUp, M_2, FillUp],axis=1)
    poss = []
    for off in range(2*y1-1):
        M = np.concatenate([np.zeros([x1, off]), M_1, np.zeros([x1, 2*y1-2-off])],axis=1)
        d = (np.linalg.norm((M-N), ord=order)**2) #/ (np.linalg.norm(M_1.flatten(),ord=0) * np.linalg.norm(M_2.flatten(),ord=0)) 
        poss.append(d)
    minimum = min(poss)
    offset  = np.argmin(poss) - y1+1
    
    
    return minimum, offset

### called by KPartite
def Total_Cost(Datenset):
    sets = len(Datenset)
    pattern = len(Datenset[0])
    Vec = Pairs(sets, 2) 
    Total_Matching_Cost = [sum(difference(Datenset[V[0]][i], Datenset[V[1]][i])[0] for V in Vec) for i in range(pattern)]    
    Total_Matching_Cost = [i/ len(Vec)/ pattern for i in Total_Matching_Cost]
    Total = round(sum(Total_Matching_Cost),3)
    
    return Total
    
### called by Compare_Trials_Pairwise
def KPartite(Data):
    '''  
    Comparison of K trials (here only called with K = 2)
    each ensemble within a trial as compared with each ensembles of each other trial
    and their Total_Cost (difference) is stored in a K dimensional matrix
    
    Parameters
    -----------
    Data : list of trials
    
    Returns
    -----------
    Matrix  : K dimensional matrix
    '''
    
    form = [len(Data[0]) for  i in range(len(Data))]
    dim  = len(form)
    Matrix = np.zeros(shape = form)
    for vec in list( it.product( range(form[0]) , repeat= dim)):
        Matrix[vec] = Total_Cost( [[Data[i][vec[i]]] for i in range( dim )])
    return Matrix

### called by create_sorted_copy
def Compare_Trials_Pairwise(Data):  
    '''  
    This function compares each trial to each other 
    regarding the similarity of their ensembles.
    This similarity is obtained by the Munkres-Kuhn method
        see Bachelor-Thesis for detailed desciption of the algorithm 
    the implementation in KuhnMunkres.py which is called with mk.Solve(Match)
    returns the cost of a match and their optimal alignment

    Parameters
    -----------   
    Data : a 3D-matrix with dimensions trials x neurons x length
     
    Returns
    -----------
    Two_Pair_Matches  : a list with all "trials choose 2" matches
    '''
    
    pairs   = Pairs(len(Data), 2)
    Two_Pair_Matches = list()
    
    for pair in pairs:
        Match = KPartite( [ Data[pair[0]], Data[pair[1]] ] )
        Two_Pair_Matches.append( [ mk.Solve(Match), pair ] )
        ### with "pair" the indices are added
    Two_Pair_Matches.sort()
    ### sort by minimum cost (gained as return of mk.Solve(Match) )
    
    return Two_Pair_Matches

def create_sorted_copy(folder): 
    '''  
    This function creates a folder containing the sorted trials, such 
    that all first ensembles match together and all seconds and so on...
    The method used is a greedy solution of k-partite matching:
        see Bachelor-Thesis for detailed desciption of the algorithm
    
    Parameters
    ----------- 
    folder : containing at least two *.h5 files with 
    
    Returns
    -----------
    ReOrder             : is a list with all information of the new order
                          [ [index of trial, [sublist: new order of ensembles]], [....,[]], ... ]
    Ensembles_sorted  : folder with same data, but each trial is sorted

    '''
    
    data = hf.load_ensembles(folder)
    
    ### compares all trials to each other - starts by aligning the best match in First = All[0]
    All = Compare_Trials_Pairwise(data) 
    
    ReOrder = list()
    used = list()
    left = set(range(len(data)))
    
    First = All[0]
    used.append(First[1][0])
    left.discard(First[1][0])
    ReOrder.append( [ First[1][0], [i[0] for i in First[0][1]] ])
    
    while left:
        
        SubAll = [i for i in All if len( set(i[1]) - set(used) ) == 1 ]  
        Next   = SubAll[0]
        New    = list(set(Next[1]).intersection(left))[0]
        Old    = list(set(Next[1]).intersection(set(used)))[0]
        pos_n  = Next[1].index(New)
        num_o  = used.index(Old)
        
        used.append(New)
        left.discard(New)
        
        sequence_new = list( i[pos_n] for i in Next[0][1] )
        sequence_old = ReOrder[num_o][1]
        
        if pos_n == 0:
            sequence_rev = [ i[1] for i in Next[0][1] ]
            sequence_new = [ sequence_new[sequence_rev.index(x)] for x in sequence_new]

        ReOrder.append( [New, [sequence_new[x] for x in sequence_old] ])
        # ReOrder is a list:[ [0, [0, 1, 2, 3]], [1, [3, 1, 2, 0]], 
        #         [3, [0, 3, 1, 2]], [2, [2, 1, 3, 0]], [4, [3, 1, 0, 2]]]
        
    ### will create Ensembles_sorted
    hf.save_Sorted(folder, ReOrder)
        