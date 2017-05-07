# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:31:50 2016

@author: ElkeKi
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

def calc_activity(spikes, ensemble):
    
    activity = np.zeros(spikes.shape[1])

    for i_frame in range(spikes.shape[1]):
        for i_lag in range(ensemble.shape[1]):
            if i_frame+i_lag >= spikes.shape[1]:
                continue

            frame1 = spikes[:,i_frame+i_lag]
            frame2 = ensemble[:,i_lag]
            activity[i_frame] += np.dot(frame1, frame2)

    return activity

def activity(dataset,dataset_name,ensemblefile,output,figurename, swap_axes):
    
    """ 
    Activity plotter
    
    
    Parameters
    -----------
        
    dataset      : the original h5 file containing the spike matrix (Note: add -dn dataset_name if the sheet is not named "spikes")
    
    dataset_name : sheet of matrix (Default: "spikes")
    
    ensemblefile : name of the .h5 file containing the identified motifs
    
    output       : name of the output file containing the calculated activities
    
    figurename   : name of the file containing the plot, for each ensemble a saparate file is created
    
    swap_axes    : transposes the input matrix
    
    """  
    
    fin1 = h5py.File(dataset,'r')
    X = fin1[dataset_name][...] 
    X = X.astype(float)
    # X contains the spikes
    if swap_axes:
        X = X.T
    
    fin2 = h5py.File(ensemblefile,'r')
    ensembles = fin2['final_motifs'][...]

    fin1.close()
    fin2.close()
    
    fout = h5py.File(output,'w')
        
    for i in range(ensembles.shape[0]):
    
        activity = calc_activity(X, ensembles[i])
        fout.create_dataset('activity_ensemble_'+str(i),data=activity)
        # calc done for ensemble i

        if np.max(activity) > 0:                  
            f = plt.figure()
            a = f.add_axes((.03,.2,.7,.35))
            a.plot(activity)
            a.set_xlim(0,X.shape[1])
            numb = i+1
            a.set_title("pattern %d" % numb)
            a.set_xlabel("frame")
            a.set_ylabel("activity")
            f.savefig(figurename+str(i)+'.png', bbox_inches='tight')

    fout.close()


