import numpy as np
import scipy.sparse as sparse
import sklearn.linear_model as sklin
import sys
from scipy import ndimage
import math
import time

class EnsembleLearner(object):
    def __init__(self):
        self.iter_ = 0

    def compute_matrix(self, data, spikes):
        
        vectorized_ensemble_len = self.n_neurons * self.n_ensembles * self.max_lag

        A = sparse.dok_matrix((self.n_neurons * self.n_frames, vectorized_ensemble_len), dtype=np.float)
        b = np.zeros((self.n_neurons * self.n_frames), order = 'F', dtype=np.float)
        
        for i_ensemble in range(self.n_ensembles):
                
            block = np.zeros((self.n_frames,self.max_lag))
            ensemble_spikes = spikes[i_ensemble, :]
                    
            for i_lag in range(self.max_lag):
                if i_lag == 0:
                    block[i_lag:,i_lag] = ensemble_spikes[:]
                else:
                    block[i_lag:,i_lag] = ensemble_spikes[:-i_lag]
            
        
            for i_neuron in range(self.n_neurons):
                sys.stdout.write("         building matrix for ensemble %d, neuron: %03d/%03d\r" % (i_ensemble, i_neuron, self.n_neurons))
                sys.stdout.flush()
                
                b_start = i_neuron * self.n_frames
                b_end = b_start + self.n_frames
                b[b_start:b_end] = data[i_neuron, :]

                row_offset = i_neuron * self.n_frames
                column_offset = i_neuron * self.max_lag + i_ensemble * self.n_neurons * self.max_lag
                
                A[row_offset : row_offset + self.n_frames , column_offset : column_offset + self.max_lag] = block
                
        
        sys.stdout.write("         building matrix: done!       \n")
        sys.stdout.flush()
        
        return A, b

    def compute_lgs(self, data, spikes):

        vectorized_ensemble_len = self.n_neurons * self.n_ensembles * self.max_lag

        A = sparse.dok_matrix((self.n_neurons * self.n_frames, vectorized_ensemble_len), dtype=np.float)
        b = np.zeros((self.n_neurons * self.n_frames), order = 'F', dtype=np.float)
        
        for i_neuron in range(self.n_neurons):
            sys.stdout.write("         building matrix for neuron: %03d/%03d\r" % (i_neuron, self.n_neurons))
            sys.stdout.flush()
            b_start = i_neuron * self.n_frames
            b_end = b_start + self.n_frames
            b[b_start:b_end] = data[i_neuron, :]

            A_row_idx = i_neuron * self.n_frames

            for i_frame in range(self.n_frames):
                offset = i_neuron

                for i_ensemble in range(self.n_ensembles):
                    ensemble_spikes = spikes[i_ensemble, :]
                    offset = i_ensemble * self.n_neurons * self.max_lag + i_neuron * self.max_lag

                    for i_lag in range(self.max_lag):
                        i_frame_real = i_frame - i_lag
                        if i_frame_real < 0:
                            continue

                        A[A_row_idx + i_frame, offset + i_lag] += ensemble_spikes[i_frame_real]
                        
        sys.stdout.write("         building matrix: done!       \n")
        sys.stdout.flush()
        return A, b

    def unravel_ensembles(self, x):
        ensembles = np.zeros((self.n_ensembles, self.n_neurons, self.max_lag))
        offset = 0
        offset_skip = self.n_neurons*self.max_lag

        for i_ensemble in range(self.n_ensembles):
            ensemble = x[offset:offset+offset_skip].reshape((self.n_neurons, self.max_lag))

            empty = []
            frames = np.unique(np.where(ensemble > 0)[1])

            for i_frame in range(self.max_lag):
                if i_frame in frames: empty.append(False)
                else: empty.append(True)

            if len(empty) > 3:
                if empty[0] == False and empty[-2] == True and empty[-1] == True:
                    # move assembly left by one
                    tmp = np.zeros((self.n_neurons, self.max_lag))
                    tmp[:,1:] = ensemble[:,:-1]
                    ensemble = tmp
                if empty[0] == True and empty[1] == True and empty[-1] == False:
                    # move assembly right by one
                    tmp = np.zeros((self.n_neurons, self.max_lag))
                    tmp[:,:-1] = ensemble[:,1:]
                    ensemble = tmp

            norm = np.linalg.norm(ensemble)
            if norm > 0:
                ensemble /= norm

            ensembles[i_ensemble, :, :] = ensemble

            offset += offset_skip

        return ensembles
     
    ######## This part centers the ensembles around their "center of mass"   #############
    
    # calculate the center of mass of the ensemble
    def cm(self,ensemble):
        cm = ndimage.measurements.center_of_mass(ensemble)
        cm_frame = cm[1]
        if math.isnan(cm_frame):
            cm_frame = int(ensemble.shape[1]/2)
        # center-of-mass-frame must be integer and we want to round correctly:
        if cm_frame-int(cm_frame) < 0.5:
            cm_frame = int(cm_frame)
        else:
            cm_frame = int(cm_frame)+1
        return cm_frame
    
    
    def unravel_ensembles_cm(self, x): 
        """
        This function centers the ensembles around its center of mass. The canter of mass is calculated by the maximum intensity of spikes within an ensemble.
    
        Note: At the moment ensembles are centered according to the number of empty columns on the left and right of the ensemble. To switch to this method change to unravel_ensembles_cm in the EnsembleLearnerLASSO class.
    
        
        Parameters
        ----------
        x :  solution of the linear system; contains coefficents of ensembles
        
        Returns
        -------
        ensembles : centered ensembles
                
        """
        
        ensembles = np.zeros((self.n_ensembles, self.n_neurons, self.max_lag))
        offset = 0
        offset_skip = self.n_neurons*self.max_lag
    
        # transforms each vector with length L = (#Neurons)*(Length of Ensemble) into a matrix with dimensions: (#Neuronen) by (Length of Ensembles)
        for i_ensemble in range(self.n_ensembles):
            ensemble = x[offset:offset+offset_skip].reshape((self.n_neurons, self.max_lag)) 
            
            # center the ensemble around the center of mass
            counter = 0
            pos_counter = 0
            while abs(self.cm(ensemble) - int(self.max_lag/2)) > 1 and counter < self.max_lag:
                if self.cm(ensemble) > int((self.max_lag)/2):
                    # move ensemble to the left by one
                    tmp = np.zeros((self.n_neurons, self.max_lag))
                    tmp[:,:-1] = ensemble[:,1:]
                    ensemble = tmp
                    pos_counter -= 1
                    print('pos',pos_counter)
                if self.cm(ensemble) < int((self.max_lag)/2):
                    # move ensemble to the right by one 
                    tmp = np.zeros((self.n_neurons, self.max_lag))
                    tmp[:,1:] = ensemble[:,:-1]
                    ensemble = tmp    
                    pos_counter += 1
                    print('pos',pos_counter)
                counter += 1
            print('c',counter)
                
            norm = np.linalg.norm(ensemble)
            if norm > 0:
                ensemble /= norm

            ensembles[i_ensemble, :, :] = ensemble
            offset += offset_skip

        return ensembles
    
    #######################################################################
    
    def set_iter(self, i):
        self.iter_ = i

class EnsembleLearnerLASSO(EnsembleLearner):
    def __init__(self, ensemble_penalty, n_ensembles, n_neurons, n_frames, max_lag):
        EnsembleLearner.__init__(self)

        self.ensemble_penalty = ensemble_penalty
        self.max_lag = max_lag
        self.n_ensembles = n_ensembles
        self.n_frames = n_frames
        self.n_neurons = n_neurons

    def learn(self, data, spikes):
        n_neurons, n_frames = data.shape
        n_ensembles, dummy = spikes.shape

        for i in range(spikes.shape[0]):
            if np.sum(spikes[i,:]) <= 0.001:
                rnd = np.random.randn(self.n_frames)
                rnd[np.where(rnd < 0)] = 0
                rnd[np.where(rnd > 0)] = 1
                spikes[i,:] = rnd

        iters = self.iter_
        
        alpha = self.ensemble_penalty
            
        A, b = self.compute_matrix(data, spikes)
        A = sparse.coo_matrix(A)
        #lasso = sklin.ElasticNet(alpha = alpha, l1_ratio=0.7, positive = True, fit_intercept = True, selection = 'random')
        lasso = sklin.Lasso(alpha = alpha, positive = True, fit_intercept = True, copy_X = False)
        lasso.fit(A, b)

        res = self.unravel_ensembles(lasso.coef_)
        # uncomment to use centering of the ensembles at the center of mass
        # res = self.unravel_ensembles_cm(lasso.coef_)
            
        return res
        
