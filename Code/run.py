from __future__ import print_function, absolute_import, division, unicode_literals

import argparse
import sys, os
import shutil
import h5py as h5

import Convensembles as learn
import Plotting as plot
import Sorting as sort

  
# Input parameters for the ensemble learning
if __name__ == "__main__":
    """ 
    Sparse convolutional coding neuronal ensemble learner
    
    This algorithm learns neuronal ensembles from a spike matrix Y of the form Y = (n_neurons, n_frames)
    Further options provide the sorting of the found ensembles and the identification of real ensembles by non-parametric significance tests. 
    """
    
    
    #Parameters
    #-----------
        
    dataset = '../../data_file_name'    # path to the original *.h5 file (without the ending '.h5') 
    
    dataset_name = 'spikes'             # dataset within the *.h5 file containing the spike matrix (Default: "spikes")
    
    folder = 'folder_name'              # name of the output folder. This folder is created in the directory where the dataset is located.  
    
    
        
    #Optional parameters for learning
    #------------
    
    quiet = False           # set True if no pictures should be generated (default: False)
    
    swap_axes = False       # if set to True, the input matrix from the *.h5 file is transposed - neccessary if the input matrix has the shape n_frames x n_neurons instead of the required n_neurons x n_frames
                            # You can check if your input matrix has the correct shape by starting the method. In the output line "building lgs for neuron: " the number of neurons should be counted through. If here the number of frames you have is shown, you have to use the swap-axes option to perform the analysis correctly.
    
    ensembles = 10          # max number of ensembles to be found (default: 10)
    
    iterations = 10         # number of iterations in each initialization (default: 10)

    length = 10             # max length of ensembles (default: 10)

    ensemble_penalty = 0.0001 # weight on the l1 norm of the motifs, the bigger this number, the sparser the learned motifs will get (default: 0.0001)

    start = 0               # frame number from which the analysis is started, useful if only poarts of the data should be analyzed (default: 0)
    
    limit = -1              # frame number up to which the analysis is performed, useful if only parts of the data should be analyzed (default: -1 for no limit)
    
    remove = None             # removes neurons from spike matrix. Enter it as a comma seperated string, e.g.: '2,3,4' (default: None)

    store_iterations = False # stores the result of each iteration 

    warm_start_file = None 
    warm_start_group = None  # name of the .h5 file and group within this file that contain values for the ensembles' activations that should be used for initialization 
    

    #Optional parameters for sorting
    #------------
    
    initializations = 5     # number of random initializations, for each trail the same set of parameters is used. If '-init 1' the sorting and non-parametric threshold estimation are not performed (default: 5)
    
    only_sort = False       # in case the ensembles have already been learned and only the sorting shall be performed (default: False) 
                            #    (NOTE: the parameter 'folder' hereby denotes the name of the folder that contains the already learned ensembles,
                            #     folder MUST contain: '/Ensembles' and '/Ensembles_random'
                            
    members = 2             # number of representatives of an ensemble that have to be similar to keep the ensemble as real ensemble (default: 2)
    
    """
    Returns
    -----------
    
    output  : saves learned ensembles in folder/Ensembles
    
    output  : saves learned ensembles from shuffled matrix in folder/Ensembles_random
    
    output  : (if initializations > 1) saves motifs after sorting in folder/Ensembles_sorted 
    
    output  : (if initializations > 1) saves motifs from the shuffled matrix after sorting in folder/Ensembles_random_sorted 

    output  : (if initializations > 1) saves motifs after the non-parametric tests in folder/Final_Motifs

    output  : (if initializations > 1) saves the shuffled spike matrix created from 'dataset.h5' in the file 'dataset_random.h5'

    """
    
    new = sort.NEW_folder
    FILES = sort.files

    def picture(folder, q):
        if not q:
            print("Generating pictures in", folder)
            for i in FILES(folder):
                plot.plot_ensembles(os.path.join(folder,i))

    

    if not dataset:
        print( "Error: No dataset given.")
        print( "Error: No dataset given.")
        sys.exit()
    elif not folder:
        print( "Error: No output folder given.")
        print( "Error: No output folder given.")
        sys.exit()
    else:
        file_name   = os.path.basename(dataset) 
        path        = os.path.dirname(dataset)
        folder      = os.path.join(path,folder)
        destination = os.path.join(folder, "Final_Motifs") 
        welcome = "\n--- Welcome to neuronal ensemble learner ---\n(results are saved in "+folder+")\n"
        sys.stdout.write(welcome)
        sys.stdout.flush()
        
    # ensembles have been learned. Loading them and skipping next part
    if only_sort: 
        if os.path.exists( os.path.join(folder, "Ensembles")) and os.path.exists(os.path.join(folder , "Ensembles_random")):
            print("\n--- Only sorts the folder",folder,"---")
            x=1
            while os.path.exists(destination):
                destination = os.path.join(folder, "Final_Motifs"  + str(x))
                x+=1
        else:
            print( "Error: Path to the files containing the ensembles not found.")
            print( "Error: Path to the files containing the ensembles not found. Make sure 'folder/Ensembles' and 'folder/Ensembles_random' exist and the path for the folder is given correctly.")
            sys.exit()
    # no ensembles have been learned yet. For doing so, create folder:        
    else:
        new(folder)
        nfolder = os.path.join(folder,'Ensembles')
        # learns the ensembles in original data and generates pictures
        learn.convolution(dataset, nfolder, dataset_name,\
                        file_name, swap_axes,\
                        ensembles,iterations,length,\
                        ensemble_penalty,\
                        limit,start,remove,\
                        initializations,store_iterations,\
                        warm_start_file,warm_start_group,quiet)
                        
        print('learning ensembles on real data is done                 \n')
    
        # searching for random ensembles if stated so 
        if initializations > 1:
      
            if not os.path.exists(dataset + '_random.h5'):
                sort.mix_matrix(dataset,dataset_name)   
                
            # learns the ensembles in randomly mixed data and generated pictures
            nfolder = os.path.join(folder,'Ensembles_random')
            learn.convolution(dataset+'_random', nfolder,\
                            dataset_name, file_name+'_random',\
                            swap_axes,\
                            ensembles,iterations,length,\
                            ensemble_penalty,\
                            limit,start,remove,\
                            initializations,store_iterations,\
                            None,None,quiet)
                            
            print('learning ensembles on random data is done            \n')
    
    # will store the final motifs in destination
    # sorting process will be started
    if only_sort or initializations > 1:

        new( os.path.join(folder,"Trash"))
        for clean in FILES(folder):
            #print("Error: The path",folder,"can't contain *.h5 files (",clean,"etc.) All *.h5 files will be moved to Trash and eventually deleted.\nPress Enter to proceed, or quit by closing console.")
            #input()
            shutil.move( os.path.join(folder,clean+".h5"), os.path.join(folder,"Trash",clean+".h5"))
            
        # finds the correct value to distinguish whether two ensembles are similar or different (threshold value)
        if members < 1:
            print("\nError: the minimum number of representatives is 1 (the medoid itsself). Please use a larger number.")
            sys.exit()            
        
        T = sort.find_threshold(folder,'Ensembles_random', 
                           members, quiet=quiet)
        if T <= 0:
            print("\nError: no appropriate threshold found or given. ")
            sys.exit()
                                   
        picture( os.path.join(folder, "Ensembles_random_sorted"),quiet)
        
        ### remove comment if folder should be deleted
        #shutil.rmtree( os.path.join(folder, "Ensembles_random_sorted"))
            
        # sort the ensembles using the found T-value
        evaluation, average, motifs = sort.run_motif_search(folder, 'Ensembles', 
                                                           T, members, 
                                                           quiet = quiet)
                                                           
        picture( os.path.join(folder, "Ensembles_sorted"),quiet)
            
        new(destination)
        fout = h5.File( destination + '/final_motifs.h5','w')
        fout.create_dataset('final_motifs', data=average)
        resgrp = fout.require_group('/candidates')        
        for i in range(len(motifs)):
            resgrp.create_dataset('candidate'+str(i), data=motifs[i]) 
        fout.close()
            
        f_content = "\nNo log file has been given in learned data.\n"
        if os.path.exists(os.path.join(folder, "Ensembles", "log.txt")):
            f = open( os.path.join(folder, "Ensembles", "log.txt"), "r")
            f_content = f.read()
            f.close()
        ev = open( os.path.join(destination,"Evaluation.txt"), "a")
        lines_of_text = ["--- finding motifs ---\n", 
                             "Dataset: "+dataset+".h5",
                             "Folder: " +folder,
                             "Sheet: " +dataset_name,
                             "H5-file: "+file_name,
                             "To: "+destination,
                             "only_sort: "+str(only_sort),
                             "Used threshold: "+str(T),
                             "Members: "+str(members)]
        ev.write('\n'.join(lines_of_text))
        for i in range(len(evaluation)):
            a = evaluation[i]
            ev.write("\n\nEnsemble "+str(i)+" (remaining candidates, neurons, motif-length): "+ str( a[0])+ " and intensity: " +str( round(a[1],3)))
        ev.write("\nEnsembles where less representatives are left then stated in Members were discarded and therfore shown as empty.\n")
        ev.write(f_content)
        ev.close()
            
        if not quiet:
            finh5 = os.path.join(destination,"final_motifs")
            plot.plot_ensembles( finh5, 
                                   dataset='final_motifs')  
            for i in range(len(motifs)):
                plot.plot_ensembles( finh5, 
                                   dataset='/candidates/candidate'+str(i),
                                   naming   ='_candidates_'+str(i))
                
            if os.path.exists(dataset + '.h5'):
                print("\n--- Now generates pictures of the activities ---")
                print("(compared to file:",dataset,"and the sheet:",dataset_name,")\n")
                plot.activity(dataset+'.h5',dataset_name, 
                            finh5+".h5", 
                            os.path.join(destination,"activities.h5"), 
                            os.path.join(destination,"activity_motif_"),
                            swap_axes)
            else:
                print( "Error: Could not find original dataset in",dataset)
                print("No activities were plotted.\n")

            print("All pictures were plotted succesfully. See the folders Ensembles, Ensembles_Random and Ensembles_sorted for different steps of the process.")
            print("The Final Motifs are shown in",destination,"\b/final_motifs.png.")
                                      
        for clean in FILES(folder):
            shutil.move(os.path.join(folder, clean+".h5"), os.path.join(folder ,"Trash",clean+".h5"))
        shutil.rmtree(os.path.join(folder,"Trash"))
        
        
    print('\n--- All done. ---\nThanks for using the neuronal ensemble learner.')
