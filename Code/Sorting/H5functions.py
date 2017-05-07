import numpy as np
np.core.arrayprint._line_width = 120
import h5py as h5

import shutil
import os
import os.path as op

######################## basic H5 functions ####################################################

### returns all H5 files in mypath
def files(mypath):
    onlyfiles = [f for f in os.listdir(mypath) if op.isfile(op.join(mypath, f))]
    h5files   = [op.splitext(f)[0] for f in onlyfiles          if op.splitext(f)[1] == '.h5']
    h5files.sort()
    return h5files

### called from main.py: creates new folder and deletes exsting if so
def NEW_folder(folder):
    if op.isdir(folder):
        shutil.rmtree(folder )
    os.makedirs(folder)

### called from mix_matrix: Reads the matrix from H5-file and returns it
def mat_from_file(file, name="result/ensembles"):
    Dataset = h5.File(file + '.h5', 'r')
    dset = Dataset[name][:]
    Dataset.close()
    return dset

### called from mix_matrix: Stores a given input matrix "daten" into a H5 file
def new_set(file, daten, name= "result/ensembles"):
    New  = h5.File(file + '.h5')
    New.create_dataset(name, data= daten)
    New.close()
     


############################# definitions used for Sorting ########################################


def compare_ensembles_at_same_position(folder):
    """
    Returns a 3D-numpy arrays containg the cross section of all trials in the folder


    Parameters
    ----------
    folder    : is the input folder, in which the sorted trials are stored


    Returns
    ----------
    Ensembles : list with 3D numpy arrays of dimemsions trials x neurons x length
                a list entry for each ensemble (firsts, senconds, thirds...)
    """
    
    Files = files(folder)
    Files.sort()
    
    Layers    = []
    Ensembles = []
    
    for i in range(len(Files)):
        ensembles = mat_from_file( op.join(folder, Files[i]), 'result/ensembles')
        
        for ens in range(len(ensembles)):

            if ens >= len(Layers):
                Layers.append([])
            Layers[ens].append(ensembles[ens])  

    for i in range(len(Layers)):
        ensembles = np.stack(Layers[i])
        Ensembles.append(ensembles)
        
    return Ensembles
   
   
### fetches all data from the name_init.h5-files in the folder
def load_ensembles(folder):

    output = list()
    Files =  files(folder)

    for trial in Files:
        data = mat_from_file( op.join(folder , trial))
        output.append( [] )

        for ensemble in range(len(data)):
            output[-1].append(data[ensemble,:,:])
    
    return output
    
    
### copies a given folder with the new order given in resort
def save_Sorted(folder, resort):
    """
    Saves the trials in a new folder, each in its correct order

    Parameters
    ----------
    folder    : the folder that was sorted
    
    resort    : the list ReOrder from the greedy method in sort_trials.py (create_sorted_copy)


    Returned *.h5 files
    ----------
    temporaryEnsembles : a temporary h5-Files, in which the sorted ensembles are storeds as well
                          this file will be deleted, after the function Find_Motifs received the data
    
    ensembles_i        : the cross section
    
    resort_trials_i    : the sorted trials
    
    """
    
    NEW_folder(folder+ "_sorted")
    Files =  files(folder)
    
    for i in resort:
        data = mat_from_file(op.join(folder, Files[i[0]]))
        shape = data.shape
        
        trials = []
        for order in i[1]:
            trials.append( data[order,:,:] )
        
        ###creates h5-Files with sorted trials to resort_name_i.h5
        path_to_res_file = op.join(folder + "_sorted", Files[i[0]]+"_sorted")
        res_data         = np.reshape(np.concatenate(trials), (shape))
        new_set(path_to_res_file, res_data)

    ### creates h5-Files with the cross section of trials to ensembles_i.h5
    Ensemble_block   = compare_ensembles_at_same_position(folder+"_sorted")
    for layer in range(len(Ensemble_block)):
        new_set( op.join(folder + "_sorted", "ensemble_"+str(layer)), 
                Ensemble_block[layer])
    
    ###creates temporary file with all information
    path_to_ens_file = op.dirname(folder)    
    new_set( op.join(path_to_ens_file,"temporaryEnsembles"), 
                Ensemble_block)
        
        
        
################################    defintions used for motif_search    #############################

def mix_matrix(dataset,dataset_name):   
    """
    Shuffles the x-axis of a matrix and stores it in H5 file.

    Parameters
    ----------
    dataset : is the input H5 file
    
    dataset_name : is the sheet name within the file


    Returns
    ----------
    The shuffled output matrix is stored in ...dataset_random.h5/dataset_name_random
    """

    A = mat_from_file(dataset, dataset_name)
    
    for neuron in range(A.shape[0]):
        tmp = A[neuron,:]
        np.random.shuffle(tmp)
        A[neuron,:] = tmp
    
    new_set(dataset+"_random",  A, dataset_name)