--------------------------------------------------------------------------------------------------------------------
    Sparse Convolutional Coding Neuronal Ensemble Learner
    
    This algorithm learns neuronal ensembles from a spike matrix Y of the form Y = (n_neurons, n_frames)
    Further options provide the sorting of the found ensembles and the elimination of spurious motifs by using a non-parametric threshold estimation. 
    
This program is written for Python 3.
However, it is possible to use it also with Python 2.7. 
We recommend using Python 3 though, as most outputs have been optimized for this version.
To have maximum speed please start directly from the machine and avoid surroundings like miniconda etc.

You get the necessary modules by adding to your command line:
pip install scipy
pip install numpy
pip install h5py
pip install matplotlib
pip install scikit-learn
pip install future (in case Python 2.7 is used)

Contact us for further questions, bug reports and other recommendations and feedback.

--------------------------------------------------------------------------------------------------------------------

    Quick Usage Guide:
    -------------

    0. Create a *.h5 File containing your neuronal data.
       (the sheet within the file should be named 'spikes')
    1. Navigate in the command line to the neuronal_ensemble_learner directory 
    2. Enter '>> python ./Code/main.py -d dataset -f folder'
    The code will start learning the ensembles from the file 'dataset.h5' with the default settings for the parameters. A folder 'folder' is created in the directory where 'dataset.h5' is located where you can find the results.
    
    
    Parameters
    -----------
        
    dataset (-d)       : the original *.h5 file (Note: add -dn dataset_name if the sheet is not named "spikes")
    
    dataset_name (-dn) : sheet of matrix (Default: "spikes")
    
    folder (-f)        : name of the output folder. This folder is created in the directory where the dataset is located.  
    
        
    Optional parameters for learning
    ------------
    
    quiet (-q)         : add -q if no pictures should be generated (default: None)
    
    swap_axes (-swap)      : if entered, the input matrix from the *.h5 file is transposed - necessary if the input matrix has the shape n_frames x n_neurons instead of the required n_neurons x n_frames
                             
    
    ensembles (-e)         : max number of ensembles to be found (default: -e 10)
    
    iterations (-i)        : number of iterations in each initialization (default: -i 10)

    length (-l)            : max length of an ensemble (default: -l 10)

    ensemble-penalty (-ep) : weight on the l1 norm of the motifs, the bigger this number, the sparser the learned motifs will get (default: -ep 0.0001)

    start (-start)         : frame number from which the analysis is started, useful if only poart of the data should be analyzed (default: -start 0)
    
    limit (-limit)         : frame number up to which the analysis is performed, useful if only part of the data should be analyzed (default: '-limit -1' for no limit)
    
    remove (-r)            : removes neurons from spike matrix. Enter it as a list, e.g.: '-r 2,3,4' (default: None)

    store_iterations (-store_iterations)    :   stores the result of each iteration 

    warm_start (-warm_start_file and -warm_start_dataset)   :   name of the .h5 file and dataset that contain values for the ensembles and their activations that should be used for initialization 
    

    Optional parameters for sorting
    ------------
    
    initializations (-init) : number of random initializations, for each trail the same set of parameters is used. If '-init 1' the sorting and non-parametric threshold estimation are not performed (default: '-init 5')
    
    only_sort (-only_sort)  : in case the ensembles have already been learned and only the sorting shall be performed (default: None) 
                            (NOTE: the parameter '-f folder' hereby denotes the name of the folder that contains the already learned ensembles,
                            folder MUST contain: '/Ensembles' and '/Ensembles_Random'
                            
    members (-m)            : number of representatives of an ensemble that have to be similar to keep the ensemble as real ensemble (default: -m 2)
    
    
    Returns
    -----------
    
    output  : saves learned ensembles in folder/Ensembles
    
    output  : saves learned ensembles from shuffled matrix in folder/Ensembles_random
    
    output  : (if initializations > 1) saves motifs after sorting in folder/Ensembles_sorted 
    
    output  : (if initializations > 1) saves motifs from the shuffled matrix after sorting in folder/Ensembles_random_sorted 

    output  : (if initializations > 1) saves motifs repeatedly appearing in different runs and their activity in folder/Final_Motifs

    output  : (if initializations > 1) saves the shuffled spike matrix created from 'dataset.h5' in the file 'dataset_random.h5'

---------------------------------------------------------------------------------------------------------
Structure in the Output-Folder 'folder'

Ensembles:
    the ensembles learned from the original data: 
    *.h5 file for each initialization, named after the input-dataset
    *.png picture corresponding to each *.h5 file
    log.txt file with all parameters and the reconstruction error

Ensembles_random:
    the ensembles learned on the shuffled data 
    *.h5 file for each initialization
    *.png picture corresponding to each *.h5 file
    log.txt file with all parameters and the reconstruction error

Ensembles_sorted:
    the ensembles in new order such that all 1st ensembles of each initialization match best, and all 2nd and all 3rd...
    to compare the trials a file with all 1st (and 2nd, 3rd...) ensembles
    *.png picture for each file

Ensembles_random_sorted:
    the ensembles from the shuffled matrix in new order such that all 1st ensembles of each initialization match best, and all 2nd and all 3rd... and
    to compare the trials a file with all 1st (and 2nd, 3rd...) ensembles
    *.png picture for each file

Final_Motifs:  
	final_motifs.h5  : this file contains the final motifs after the motifs that appeared in different runs in one sheet
               and in a group called 'candidates' one finds all the remaining representatives of each ensemble, from which the final motifs were computed by taking the minimal spike values
	activities.h5   : temporal occurrence of each motif in the original data 
	Evaluation.txt  : a file with all parameters
    	final_motifs.png: picture of all final motifs 
	*.png pictures containing all the candidates for each ensemble
	*.png picture for the activity of each motif
	(if this folder exists, a new folder Final_Motifs_1 will be created if the sorting is performed a second time and so on)

All pictures can be turned off with (-q).

--------------------------------------------------------------------------
Usage of store_iterations and warm_start:

In some cases (especially when analysing large datasets where the computations 
are very time consuming) it might be reasonable to save the results of each single 
iteration. This can be done by adding "-store_iterations" to the comand line when starting 
the ensemble learner. The ensembles and their activations found in each iteration 
will be saved in the output *.h5 file in a seperate group (named "iter_1", "iter_2", ...). 

To restart the analysis from a previously stored stage, the warm-start option can be used. 
By adding a warm-start-file and -dataset (by adding "-warm_start_file filename 
-warm_start_dataset datasetname") the ensembles and their activations are 
initialized from the ones stored in that dataset. 

If you use multiple initializations together with the warm-start option, all of them 
will be initialized from the warm-start-dataset. 
This option overwrites the usual random initialization of the activations. 
The randomness in the initialization is crucial for the sorting algorithm.
Therefore make sure to only use this option if you want to start from a specific setting, 
e.g. to continue some analysis that had been interrupted. 

If the warm-start option is used to continue an interrupted analysis, the folder-name
has to be changed. Otherwise the original output file will be overwritten before 
the ensembles and activations can be loaded.


Example:
Start original analysis:
>> python ./Code/main.py -d dataset -f folder -i 10 -store_iterations
If the analysis has to be stopped after e.g. 6 iterations, the analysis can be restarted by:
>> python ./Code/main.py -d dataset -f folder_continued -i 4 -warm_start_file folder/Ensembles/dataset_0 -warm_start_dataset iter_6
 

--------------------------------------------------------------------------
Known bugs and how to avoid them:


Bug:       If no motifs could be found, there are two possible reasons: 
		a. The ensemble penalty is too strong and as a result the learned ensembles are empty.
		b. The learned ensembles are not empty but the sorting excludes them because they are too different.

Solution:   a. Decrease the ensemble penalty (parameter -ep, a smaller value will allow more spikes to be present in the motifs)
            b. Check the similarity of the found ensembles by eye and compare them manually. 
            

Bug: 	   Python2 cannot import convensembles (should not occur anymore!)
Solution:  Use Python 3.5 or later

Bug:       Python2 cannot recognize numpy.stack module
Solution:  Please use numpy version 1.11.1 or later
          ( pip install numpy --upgrade )
 

