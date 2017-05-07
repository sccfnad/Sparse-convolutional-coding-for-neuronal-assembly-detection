import numpy as np
import scipy.sparse as sparse
import sklearn.linear_model as sklin

class SpikeLearner(object):
    def __init__(self):
        pass


class SpikeLeanerPursuit(SpikeLearner):
    def __init__(self):
        SpikeLearner.__init__(self)

    def learn(self, data, ensembles):
        n_neurons, n_frames = data.shape
        n_ensembles, dummy, max_lag = ensembles.shape

        spikes = np.zeros((n_ensembles, n_frames))
        iprod = np.zeros((n_ensembles, n_frames))
        residual = np.copy(data)
        recons = np.zeros(data.shape)

        first = True
        frame = -1
        last_diff = np.Inf

        while True:
            if first:
                for i_frame in range(n_frames):
                    for i_ensemble in range(n_ensembles):
                        ensemble = ensembles[i_ensemble,:,:]
                        for i_lag in range(max_lag):
                            if i_frame + i_lag >= n_frames:
                                continue
                            iprod[i_ensemble, i_frame] += np.dot(residual[:, i_frame + i_lag], ensemble[:,i_lag])
                first = False
            else:
                for i_frame in range(n_frames):
                    if abs(frame - i_frame) > max_lag:
                        continue

                    iprod[:, i_frame] = 0.

                for i_frame in range(n_frames):
                    if abs(frame - i_frame) > max_lag:
                        continue        
                    for i_ensemble in range(n_ensembles):
                        ensemble = ensembles[i_ensemble,:,:]
                        for i_lag in range(max_lag):
                            if i_frame + i_lag >= n_frames:
                                continue
                            iprod[i_ensemble, i_frame] += np.dot(residual[:, i_frame + i_lag], ensemble[:,i_lag])


            ensemble, frame = np.unravel_index(np.argmax(iprod), iprod.shape)
            val = iprod[ensemble, frame]

            spikes[ensemble, frame] += val
            
            for i_lag in range(max_lag):
                if frame + i_lag >= n_frames:
                    continue
                
                recons[:, frame + i_lag] += val * ensembles[ensemble, :, i_lag]
                residual[:, frame + i_lag] -= val * ensembles[ensemble, :, i_lag]
                
            diff = np.sqrt(np.sum(np.power(recons - data, 2)))

            if val < 0.001 or diff > last_diff:
                spikes[ensemble, frame] -= val
                break

            last_diff = diff
        
            
        self.last_diff = diff
                
        return last_diff, spikes
