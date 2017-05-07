from .spikes import SpikeLeanerPursuit
from .ensembles import EnsembleLearnerLASSO

import sys, re, os
import h5py as h5
import numpy as np

def rewrite(lines):
    for _ in range(lines):
        sys.stdout.write("\x1b[A")
        sys.stdout.write(re.sub(r"[^\s]", " ", ""))
    for _ in range(lines):
        sys.stdout.write("                                              \n")
    for _ in range(lines):
        sys.stdout.write("\x1b[A")
        sys.stdout.write(re.sub(r"[^\s]", " ", ""))
    
def convolution(dataset,folder,dataset_name,file_name, swap_axes,\
                n_ensembles,iterations,lag,ensemble_penalty,\
                limit,start,remove,initializations,store_iterations,\
                warm_start_file,warm_start_dataset):
    ''' 
    This function solves the optimization problem 
    :math:`\min_{\mathbf{a}, \mathbf{s}} \left\| \mathbf{Y} - \sum_i^l \mathbf{s}_i \circledast \mathbf{a}_i \right\|_F^2 + \alpha \sum_i^{l} \|\mathbf{s}_i\|_0 + \beta \sum_i^l \| \mathbf{a}_i \|_1`
    by using a block coordinate descent strategy.
    
    Parameters
    -----------
        
    dataset              : the original *.h5 file 
    
    dataset_name         : sheet of matrix 
    
    folder               : name of the output folder. This folder is created in the directory where the dataset is located.  
    
    swap_axes            : if True, the input matrix from the *.h5 file is transposed
    
    n_ensembles            : max number of ensembles to be found 
    
    iterations          : number of iterations in each initialization 

    lag                 : max length of ensembles 

    ensemble-penalty    : ensemble coefficient beta, the bigger this value is, the sparser the motifs will be

    start               : frame number from which the analysis is started, useful if only poart of the data should be analyzed 
    
    limit               : frame number up to which the analysis is performed, useful if only part of the data should be analyzed 
    
    remove              : removes neurons from spike matrix 

    initializations     : number of random initializations, for each trail the same set of parameters is used. 

    store_iterations    :   stores the result of each iteration 

    warm_start_file and warm_start_dataset   :   name of the .h5 file and dataset that contain values for the ensembles and spikes that should be used for initialization 
    
    Output
    -----------
    
    learned ensembles and spikes saved in an *.h5 file
    
    '''        
    os.makedirs(folder)
    fin = h5.File(dataset+'.h5', 'r')
    spikes_mtx = fin[dataset_name][...]
    spikes_mtx = spikes_mtx.astype(float)
    fin.close()

    if swap_axes:
        spikes_mtx = np.swapaxes(spikes_mtx, 0, 1)


    if remove is not None:
        remove = [int(i) for i in remove.split(',')] #[ 1,  7, 11, 15, 26, 29, 43, 57, 58, 60, 62]
        for r in remove:
            spikes_mtx[r,:] = 0

    spikes_mtx = spikes_mtx[:,start:]
    if limit > 0:
        spikes_mtx = spikes_mtx[:,:limit]
    
        
    n_neurons, n_frames = spikes_mtx.shape
    
    
    learner_spikes = SpikeLeanerPursuit()
    learner_ensembles = EnsembleLearnerLASSO(ensemble_penalty, n_ensembles, n_neurons, n_frames, lag)

    spikes = np.zeros((n_ensembles, n_frames))
    ensembles = np.zeros((n_ensembles, n_neurons, lag))
    
    txt = os.path.join(folder,"log.txt")
    fh  = open(txt, "w")
    lines_of_text = ["--- finding ensembles ---\n", 
                     "Dataset: "+dataset+".h5",
                     "Folder: " +folder,
                     "Sheet: " +dataset_name,
                     "H5-file: "+file_name, 
                     "Swap-axes: " +str(swap_axes),
                     "Number of ensembles: " + str(n_ensembles),
                     "Number of iterations: " + str(iterations),
                     "Length of ensembles: " + str(lag),
                     "Ensembles-Penalty: " + str(ensemble_penalty),
                     "Limit: " + str(limit),
                     "Start: " + str(start),
                     "Removed neurons: " + str(remove),
                     "Number of initializations: " +str(initializations)]
    fh.write('\n'.join(lines_of_text) + '\n\nReconstruction Errors (initialization, iteration):\n\n')
    fh.close()
    
    print("\n\nThe data to be analysed consists of %d neurons observed over %d time frames. \nIf this is not correct, use the -swap option to transpose the inserted matrix.\n\n" % (n_neurons,n_frames))
    print("--- finding", n_ensembles,"ensembles with length", lag,"in", file_name,"\b.h5 --- \n(for more information see log.txt) \n\n\n\n\n\n\n")
    for init in range(initializations):
        rewrite(5)
        
        print( "initialization %02d/%02d" % (init+1, initializations))
        
        name_of_init = os.path.join(folder , file_name +'_'+str(init))
        fout = h5.File(name_of_init+'.h5', 'w')

        for i in range(n_ensembles):
            rnd = np.random.randn(n_frames)
            rnd[np.where(rnd < 0)] = 0
            rnd[np.where(rnd > 0)] = 1
            spikes[i, :] = rnd
            
        if warm_start_file:
            print('warm_start',warm_start_file)
            fw = h5.File(warm_start_file+'.h5', 'r')
            grp = fw[warm_start_dataset]
            ensembles = grp["ensembles"][:]
            spikes = grp["activations"][:]
            fw.close()


        print("\n\n\n")
        for i in range(iterations):
            rewrite(4)
            
            print( "   iteration %02d/%02d" % (i+1, iterations))
            
            print( "      learning ensembles")
            learner_ensembles.set_iter(i)
            ensembles = learner_ensembles.learn(spikes_mtx, spikes)

            print( "      learning activations")
            reco, spikes = learner_spikes.learn(spikes_mtx, ensembles)
            
            fh = open(txt, "a")
            fh.write("RE (" +str(init)+","+ str(i)+"): "+str(reco)+"\n")
            fh.close()
            
            if reco == np.Inf:
                print("ERROR: Reconstruction Error is " +str(reco)+". Decrease ensemble penalty and try again.")
                sys.exit() 
            
            if store_iterations:
                #grp_base = fout.require_group(store_iterations)
                grp_iter = fout.require_group("iter_%d" % (i+1))
                grp_iter.create_dataset("activations", data=spikes)
                grp_iter.create_dataset("ensembles", data=ensembles)
                fout.flush()


        resgrp = fout.require_group('/result')        
        resgrp.create_dataset("activations", data=spikes)
        resgrp.create_dataset("ensembles", data=ensembles)
        fout.close()
    
    fh = open(txt, "a")
    fh.write("--- finished --- ")
    fh.close()
        
        
        