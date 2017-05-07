from __future__ import print_function
import os
import numpy as np
np.core.arrayprint._line_width = 120
import shutil

import Sorting.H5functions as hf
import Sorting.sort_trials as so

    
def Find_Motifs(Ensembles, T):
    '''  
    In this function the final motifs are generated from the sorted sets. 
    
    Parameters
    -----------
    T          : threshold of the difference
    
    Ensembles  : list of 3D-matrices with matched ensembles
                 each entry has dimensions trials x neurons x length
                 there are as many entries as different ensembles
    
    Returns
    -----------
    list_of_candidate_blocks:  a list of 3D-matrices with dimensions candidates x neurons x length(shortened)
    '''
    
    list_of_candidate_blocks = []
    # Block is now a 3D numpy array with trials x neurons x length
    # counter runs through the indices of the ensemble in Block
    for counter, Block in enumerate(Ensembles):  
        
        anz = len(Block) # number of trials
        Pairs = so.Pairs(anz, 2) 
        Diffs = np.zeros(shape=(anz,anz))
        Offs  = np.zeros(shape=(anz,anz))
        # Diffs is a matrix in which the entry ij denote the difference between i-th and j-th ensemble
        
        for (p,q) in Pairs:
            M_1 = np.copy(Block[p])
            M_2 = np.copy(Block[q])
            Diffs[p,q], Offs[p,q] = so.difference(Block[p],Block[q]) 
            Diffs[p,q] /= (np.linalg.norm(M_1.flatten(),0) * np.linalg.norm(M_2.flatten(),0))
            Diffs[q,p], Offs[q,p] = so.difference(Block[q],Block[p]) 
            Diffs[q,p] /= (np.linalg.norm(M_2.flatten(),0) * np.linalg.norm(M_1.flatten(),0))
            # the ensembles from the different trials in all possible combinations
            # and the shifts that lead to the minimal difference between each pair of ensembles
        
        Sums_of_Differences = np.sum(Diffs,axis=0) + np.sum(Diffs,axis=1)
        
        Medoid = np.argmin(Sums_of_Differences)
        #print('ensemble, medoid')
        #print(counter, Medoid)
        # the medoid fits best to all other ensembles
        # now the difference to it is fetched from the Diffs-Matrix
        Difference_to_Medoid = [max(Diffs[j,Medoid], Diffs[Medoid,j]) for j in range(anz)]
        Offset_to_Medoid     = [Offs[j,Medoid] for j in range(Medoid+1)] + [-1*Offs[Medoid,i] for i in range(Medoid+1,anz)]
        #Offset_to_Medoid = [-1*Offs[0,j] for j in range(anz)]       
        
        # Offset: columns of zeros before the candidate to be aligned with the medoid        
        Candidates = [j for j in range(anz) if (T > Difference_to_Medoid[j])]
        # Candidates: list of the indices of the ensembles with a difference to the medoid lower than threshold T
        
        s = Block.shape
        Block = np.concatenate([np.zeros(s), Block, np.zeros(s) ],axis=2)
        # Block is lengthened to shift over other ensembles
        # Block has now dimension: trials x neurons x (length*3) 
        
        if not Candidates:
            # print("Attention: Threshold doesn't make sense (the ensemble",counter," has a L0-norm of",np.diagonal(Diffs)[Medoid],". However the threshold is with",T,"higher than this value, so not even the Medoid will be added to the remaining ensembles. Please remember this in further evaluation.")
            Candidates.append(Medoid)
         
        motifs = []
        # centers every candidate, so that it will be aligned with the medoid
        for c in Candidates:
            Motif = Block[c]
            off   = Offset_to_Medoid[c]
            motifs.append( Motif[:,slice( int(s[2]-off) , int(2*s[2]-off))]) 
            # keeps the original length of ensemble but with best shift with respect to the medoid
        
        list_of_candidate_blocks.append( np.stack( motifs ) )

    return list_of_candidate_blocks
    
    
def find_finals(block_of_motifs, members):
    '''  
    Up to this point all ensembles left have been compared, and those that don't fit in have been deleted.
    The remaining ensembles are called candidates and are stored in the final_motifs.h5 (also plotted as png)
    To obtain single pictures of a final motif, here the combination is built (if more than 'members' remain)
    This function will set each spike of the final motif to the minimum of all candidates.
    
    Parameters
    -----------
    block_of_motifs : list of 3D-ensemble matrices with dimension candidates x neurons x length
    
    members         : minimum number of matching ensembles
    
    
    Returns
    -----------
    evaluation  : list with an entry for each motif
                  [the number of candidates, sum of all spikes]
    
    motif-block : the motifs themselves, as a concatenated 3D matrix
    '''
    
    add         = []
    evaluation  = []
    motif_block = []
    
    for ensset in block_of_motifs:
        anz = ensset.shape 
        # use minimum:
        B = np.zeros((anz[1],anz[2]))
        if anz[0] < members:
            add.append(B)
            evaluation.append( [ anz, np.sum(B**2) ])
            continue 
        for n in range(anz[1]):
            for f in range(anz[2]):
                B[n,f] = np.min(ensset[:,n,f])
        add.append(B)
        evaluation.append( [ anz, np.sum(B**2) ])

    #equals the length of all motifs    
    maximum = max( [i[2] for i,j in evaluation] )
    for mot in add:     
        x,l = mot.shape
        M = np.concatenate([mot, np.zeros([x, maximum - l])],axis=1)
        motif_block.append(M)

    ### return two entries: evaluation and motif-block
    return evaluation, motif_block



def run_motif_search(directory, folder, threshold, members, quiet=True):
    '''  
    This function performs Non-parametrical tests on given ensembles. 
    
    Parameters
    -----------  
    directory : directory of the folders Ensembles and Ensembles_Random
    
    members   : minimum left ensembles to create a motif 
    
    threshold : maximum difference that ensembles have to have with the Medoid
    
    folder    : folder that contains the learned ensembles H5 files (with same dimensions!)
    
    quiet     : True/False value, whether to create pictures (Default False means no pictures)
    
    
    Returns
    -----------
    evaluation       : table with data about the resulting motifs
    
    final_motifs     : found motifs in 3D-numpy array
    
    candidates       : before taking the minimum spike values
    '''
    
    
    folder = os.path.join(directory , folder)
    if not quiet:        
        print("\n--- sorting ensembles in ",folder, "with greedy method ---")

    ### HERE POSSIBLE: Uncomment section, if Folder_sorted already exists
    so.create_sorted_copy(folder)
    folder_Res = folder+"_sorted"
    if not quiet:        
        print("(Sorted succesfully into", folder_Res, "\b)")
    ### UNTIL HERE

    Ensembles  = hf.mat_from_file( os.path.join(directory, "temporaryEnsembles"))
    candidates = Find_Motifs( Ensembles, T=threshold)
    evaluation, final_motifs  = find_finals(candidates, members)
    
    # final_motifs contains the result (found set of motifs from matching the computed motifs)
    return (evaluation, final_motifs,  candidates)
       
       
       
def find_threshold(directory, folder, members, quiet=True):
    
    '''  
    This function finds a threshold to distinguish between similar ensembles 
    and random ensembles. The lower the threshold is, the less ensembles
    will be found to be "similar".
    The threshold is computed as the minimal difference between an ensemble and the medoid.
    
    Parameters
    -----------
    directory : dir in which the folder (Ensembles or Ensembles_random) is

    folder    : folder that contains the learned ensembles H5 files (with same dimensions!)
    
    members   : minimum left ensembles to create a motif 
    
    
    Returns
    -----------
    threshold : threshold for difference of two ensembles
    '''
    
    print("--- finding threshold ---")
    
    # computing the threshold directly as minimum of the difference of the sorted motifs to the medoid
    # find sorted ensembles
    folder = os.path.join(directory , folder)
    if not quiet:        
        print("\n--- sorting ensembles in ",folder, "with greedy method ---")

    so.create_sorted_copy(folder)
    folder_Res = folder+"_sorted"
    if not quiet:        
        print("(Sorted succesfully into", folder_Res, "\b)")
    
    Ensembles  = hf.mat_from_file( os.path.join(directory, "temporaryEnsembles"))
    
    minimal_differences_list = []
    for counter, Block in enumerate(Ensembles):  
        
        anz = len(Block) # number of trials
        Pairs = so.Pairs(anz, 2) 
        Diffs = np.zeros(shape=(anz,anz))
        Offs  = np.zeros(shape=(anz,anz))
        # Diffs is a matrix in which the entry ij denote the difference between i-th and j-th ensemble
        
        for (p,q) in Pairs:
            M_1 = np.copy(Block[p])
            M_2 = np.copy(Block[q])
            Diffs[p,q], Offs[p,q] = so.difference(Block[p],Block[q]) 
            #if np.linalg.norm(M_1.flatten(),0) <= 0 or np.linalg.norm(M_2.flatten(),0) <= 0:
            #    continue
            Diffs[p,q] /= (np.linalg.norm(M_1.flatten(),0) * np.linalg.norm(M_2.flatten(),0))
            # the ensembles from the different trials in all possible combinations
            # and the shifts that lead to the minimal difference between each pair of ensembles
        
        Sums_of_Differences = np.sum(Diffs,axis=0) + np.sum(Diffs,axis=1)
        
        #for i in range(anz):
        #    if np.max(Block[i]) <= 0:
        #        Sums_of_Differences[i] = np.inf
        
        Medoid = np.argmin(Sums_of_Differences)
        # the medoid has the smallest difference to all other ensembles
        # now the difference to it is fetched from the Diffs-Matrix
        Difference_to_Medoid = [max(Diffs[j,Medoid], Diffs[Medoid,j]) for j in range(anz)]
        #for i in range(anz):
        #    if np.max(Block[i]) <= 0:
        #        Difference_to_Medoid[i] = np.inf
        Diff_to_Medoid_sorted = sorted(Difference_to_Medoid)
        # taking the minimal difference value for the specified number of ensembles
        # usually members=2 so T is the smallest difference of a motif -which is not the medoid itsself- to the medoid 
        T = Diff_to_Medoid_sorted[members-1] 
        minimal_differences_list.append(T)
        
    shutil.move( os.path.join(directory,"temporaryEnsembles.h5"), os.path.join(directory,"Trash", "temporaryEnsembles.h5"))

    threshold = min(minimal_differences_list)
    
    print("Threshold: ",threshold)
    
    
    return(threshold)
    
           
              
