from __future__ import print_function, absolute_import, division, unicode_literals

import argparse
import sys, os
import shutil
import h5py as h5

import Convensembles as learn
import Plotting as plot
import Sorting as sort

new = sort.NEW_folder
FILES = sort.files

def picture(folder, q):
    if not q:
        print("Generating pictures in", folder)
        for i in FILES(folder):
            plot.plot_ensembles(os.path.join(folder,i))
  
# Input parameters for the ensemble learning
if __name__ == "__main__":
    """ 
    Sparse convolutional coding neuronal ensemble learner
    
    This algorithm learns neuronal ensembles from a spike matrix Y of the form Y = (n_neurons, n_frames)
    Further options provide the sorting of the found ensembles and the identification of real ensembles by non-parametric significance tests. 
    
    
    Parameters
    -----------
        
    dataset (-d)       : the original *.h5 file (Note: add -dn dataset_name if the sheet is not named "spikes")
    
    dataset_name (-dn) : sheet of matrix (Default: "spikes")
    
    folder (-f)        : name of the output folder. This folder is created in the directory where the dataset is located.  
    
    
        
    Optional parameters for learning
    ------------
    
    quiet (-q)         : add -q if no pictures should be generated (default: None)
    
    swap_axes (-swap)      : if entered, the input matrix from the *.h5 file is transposed - neccessary if the input matrix has the shape n_frames x n_neurons instead of the required n_neurons x n_frames
                             You can check if your input matrix has the correct shape by starting the method. In the output line "building lgs for neuron: " the number of neurons should be counted through. If here the number of frames you have is shown, you have to use the swap-axes option to perform the analysis correctly.
    
    ensembles (-e)         : max number of ensembles to be found (default: -e 10)
    
    iterations (-i)        : number of iterations in each initialization (default: -i 10)

    length (-l)            : max length of ensembles (default: -l 10)

    ensemble-penalty (-ep) : weight on the l1 norm of the motifs, the bigger this number, the sparser the learned motifs will get (default: -ep 0.0001)

    start (-start)         : frame number from which the analysis is started, useful if only poarts of the data should be analyzed (default: -start 0)
    
    limit (-limit)         : frame number up to which the analysis is performed, useful if only parts of the data should be analyzed (default: '-limit -1' for no limit)
    
    remove (-r)            : removes neurons from spike matrix. Enter it as a list, e.g.: '-r 2,3,4' (default: None)

    store_iterations (-store_iterations)    :   stores the result of each iteration 

    warm_start (-warm_start_file and -warm_start_group)   :   name of the .h5 file and group within this file that contain values for the ensembles' activations that should be used for initialization 
    

    Optional parameters for sorting
    ------------
    
    initializations (-init) : number of random initializations, for each trail the same set of parameters is used. If '-init 1' the sorting and non-parametric threshold estimation tests are not performed (default: -init 5)
    
    only_sort (-only_sort)  : in case the ensembles have already been learned and only the sorting shall be performed (default: None) 
                            (NOTE: the parameter '-f folder' hereby denotes the name of the folder that contains the already learned ensembles,
                            folder MUST contain: '/Ensembles' and '/Ensembles_random'
                            
    members (-m)            : number of representatives of an ensemble that have to be similar to keep the ensemble as real ensemble (default: -m 2)
    
    
    Returns
    -----------
    
    output  : saves learned ensembles in folder/Ensembles
    
    output  : saves learned ensembles from shuffled matrix in folder/Ensembles_random
    
    output  : (if initializations > 1) saves motifs after sorting in folder/Ensembles_sorted 
    
    output  : (if initializations > 1) saves motifs from the shuffled matrix after sorting in folder/Ensembles_random_sorted 

    output  : (if initializations > 1) saves motifs after the non-parametric tests in folder/Final_Motifs

    output  : (if initializations > 1) saves the shuffled spike matrix created from 'dataset.h5' in the file 'dataset_random.h5'

    """
    
    parser = argparse.ArgumentParser(description='neuronal ensemble learner')

    parser.add_argument('-d', '--dataset', help="MANDATORY: file with the neuron spike matrix, must be .h5 file, enter file name without ending .h5")
    parser.add_argument('-dn', '--dataset-name', default="/spikes", help="path to the spike matrix in the .h5 file (default: %(default)s)")
    parser.add_argument('-f', '--folder', help="MANDATORY: name of the output folder, without slash at the end (eg: -f NEW_RESULTS")    
    
    parser.add_argument('-swap', '--swap-axes', help="transposes the spike matrix if data is stored as (n_frames, n_neurons) instead of (n_neurons, n_frames)", action="store_true")
    parser.add_argument('-e', '--ensembles', type=int, default=10, help="max number of ensembles (default: %(default)d)")
    parser.add_argument('-i', '--iterations', type=int, default=10, help="number of iterations (default: %(default)d)")
    parser.add_argument('-l', '--length', type=int, default=10, help="temporal length of ensembles (default: %(default)d")
    parser.add_argument('-ep', '--ensemble-penalty', type=float, default=0.0001, help="ensemble coefficient beta, the bigger this value is, the sparser the motifs will get (default: %(default)s)")
    parser.add_argument('-limit ', '--limit', type=int, default=-1, help="limit number of frames")
    parser.add_argument('-start',  '--start', type=int, default=0, help="frame to start from, useful in combination with --limit if only part of the data should be analyzed")
    parser.add_argument('-r', '--remove', help="remove neurons from spike matrix")
    parser.add_argument('-q', '--quiet', action='store_true', help="Add -q for no pictures. Will be faster. Create pictures later e.g. with 'python ./Code/Plotting/plot_ensembles.py -d matrix'")
 
    parser.add_argument('-init', '--initializations', type=int,default=5,help="number of random initializations with same paramters (default: %(default)d)")
    parser.add_argument('-only_sort', '--only-sort', action="store_true", help="If the ensembles have been learned already, add -only_sort to skip relearning (ATTENTION: use -f to specify folder) (Default: None)")
    parser.add_argument('-m','--members',type=float,default=2,help='minimum number of ensembles found to be equivalent to keep a motif. if less the motif is deleted (default: %(default)d)')    
    
    parser.add_argument('-store_iterations', '--store-iterations', action="store_true", help="store results of every iteration")
    parser.add_argument('-warm_start_file', '--warm-start-file', help="start with initial values for ensembles and spikes loaded from this .h5 file")
    parser.add_argument('-warm_start_group', '--warm-start-group', help="group within the warm-start-file that contains the (ensembles and) spikes to start from")

    args = parser.parse_args()
    

    if not args.dataset:
        print( "Error: No dataset given.")
        parser.print_help()
        print( "Error: No dataset given.")
        sys.exit()
    elif not args.folder:
        print( "Error: No output folder given.")
        parser.print_help()
        print( "Error: No output folder given.")
        sys.exit()
    else:
        file_name   = os.path.basename(args.dataset) 
        path        = os.path.dirname(args.dataset)
        folder      = os.path.join(path,args.folder)
        destination = os.path.join(folder, "Final_Motifs") 
        start = "\n--- Welcome to neuronal ensemble learner ---\n(results are saved in "+folder+")\n"
        sys.stdout.write(start)
        sys.stdout.flush()
        
    # ensembles have been learned. Loading them and skipping next part
    if args.only_sort: 
        if os.path.exists( os.path.join(folder, "Ensembles")) and os.path.exists(os.path.join(folder , "Ensembles_random")):
            print("\n--- Only sorts the folder",folder,"---")
            x=1
            while os.path.exists(destination):
                destination = os.path.join(folder, "Final_Motifs"  + str(x))
                x+=1
        else:
            print( "Error: Path to the files containing the ensembles not found.")
            parser.print_help()
            print( "Error: Path to the files containing the ensembles not found. Make sure 'folder/Ensembles' and 'folder/Ensembles_random' exist and the path for the folder is given correctly.")
            sys.exit()
    # no ensembles have been learned yet. For doing so, create folder:        
    else:
        new(folder)
        nfolder = os.path.join(folder,'Ensembles')
        # learns the ensembles in original data and generates pictures
        learn.convolution(args.dataset, nfolder, args.dataset_name,\
                        file_name, args.swap_axes,\
                        args.ensembles,args.iterations,args.length,\
                        args.ensemble_penalty,\
                        args.limit,args.start,args.remove,\
                        args.initializations,args.store_iterations,\
                        args.warm_start_file,args.warm_start_group,args.quiet)
                        
        print('learning ensembles on real data is done                 \n')
    
        # searching for random ensembles if stated so 
        if args.initializations > 1:
      
            if not os.path.exists(args.dataset + '_random.h5'):
                sort.mix_matrix(args.dataset,args.dataset_name)   
                
            # learns the ensembles in randomly mixed data and generated pictures
            nfolder = os.path.join(folder,'Ensembles_random')
            learn.convolution(args.dataset+'_random', nfolder,\
                            args.dataset_name, file_name+'_random',\
                            args.swap_axes,\
                            args.ensembles,args.iterations,args.length,\
                            args.ensemble_penalty,\
                            args.limit,args.start,args.remove,\
                            args.initializations,args.store_iterations,\
                            None,None,args.quiet)
                            
            print('learning ensembles on random data is done            \n')
    
    # will store the final motifs in destination
    # sorting process will be started
    if args.only_sort or args.initializations > 1:

        new( os.path.join(folder,"Trash"))
        for clean in FILES(folder):
            #print("Error: The path",folder,"can't contain *.h5 files (",clean,"etc.) All *.h5 files will be moved to Trash and eventually deleted.\nPress Enter to proceed, or quit by closing console.")
            #input()
            shutil.move( os.path.join(folder,clean+".h5"), os.path.join(folder,"Trash",clean+".h5"))
            
        # finds the correct value to distinguish whether two ensembles are similar or different (threshold value)
        if args.members < 1:
            print("\nError: the minimum number of representatives is 1 (the medoid itsself). Please use a larger number.")
            sys.exit()            
        
        T = sort.find_threshold(folder,'Ensembles_random', 
                           args.members, quiet=args.quiet)
        if T <= 0:
            print("\nError: no appropriate threshold found or given. ")
            sys.exit()
                                   
        picture( os.path.join(folder, "Ensembles_random_sorted"),args.quiet)
        
        ### remove comment if folder should be deleted
        #shutil.rmtree( os.path.join(folder, "Ensembles_random_sorted"))
            
        # sort the ensembles using the found T-value
        evaluation, average, motifs = sort.run_motif_search(folder, 'Ensembles', 
                                                           T, args.members, 
                                                           quiet = args.quiet)
                                                           
        picture( os.path.join(folder, "Ensembles_sorted"),args.quiet)
            
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
                             "Dataset: "+args.dataset+".h5",
                             "Folder: " +folder,
                             "Sheet: " +args.dataset_name,
                             "H5-file: "+file_name,
                             "To: "+destination,
                             "only_sort: "+str(args.only_sort),
                             "Used threshold: "+str(T),
                             "Members: "+str(args.members)]
        ev.write('\n'.join(lines_of_text))
        for i in range(len(evaluation)):
            a = evaluation[i]
            ev.write("\n\nEnsemble "+str(i)+" (remaining candidates, neurons, motif-length): "+ str( a[0])+ " and intensity: " +str( round(a[1],3)))
        ev.write("\nEnsembles where less representatives are left then stated in Members were discarded and therfore shown as empty.\n")
        ev.write(f_content)
        ev.close()
            
        if not args.quiet:
            finh5 = os.path.join(destination,"final_motifs")
            plot.plot_ensembles( finh5, 
                                   dataset='final_motifs')  
            for i in range(len(motifs)):
                plot.plot_ensembles( finh5, 
                                   dataset='/candidates/candidate'+str(i),
                                   naming   ='_candidates_'+str(i))
                
            if os.path.exists(args.dataset + '.h5'):
                print("\n--- Now generates pictures of the activities ---")
                print("(compared to file:",args.dataset,"and the sheet:",args.dataset_name,")\n")
                plot.activity(args.dataset+'.h5',args.dataset_name, 
                            finh5+".h5", 
                            os.path.join(destination,"activities.h5"), 
                            os.path.join(destination,"activity_motif_"),
                            args.swap_axes)
            else:
                print( "Error: Could not find original dataset in",args.dataset)
                print("No activities were plotted.\n")

            print("All pictures were plotted succesfully. See the folders Ensembles, Ensembles_Random and Ensembles_sorted for different steps of the process.")
            print("The Final Motifs are shown in",destination,"\b/final_motifs.png.")
                                      
        for clean in FILES(folder):
            shutil.move(os.path.join(folder, clean+".h5"), os.path.join(folder ,"Trash",clean+".h5"))
        shutil.rmtree(os.path.join(folder,"Trash"))
        
        
    print('\n--- All done. ---\nThanks for using the neuronal ensemble learner.')
