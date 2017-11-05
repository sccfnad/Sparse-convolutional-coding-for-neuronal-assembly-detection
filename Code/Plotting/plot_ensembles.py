#cfrom __future__ import print_function
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import argparse


def plot_ensembles(path, dataset = 'result/ensembles', naming=""):

    '''  
    This function plots the content of a h5-file and stores the picture as a png-file. 
    
    
    Parameters
    -----------
    
    path        : path to the h5-file without(!) the suffix '.h5'
    
    dataset     : the name of the dataset, that contains i ensembles (Default: 'result/ensembles')
      
    
    Returns
    -----------
    
    output_name : In case the path = './Ensembles/Data_1' so that the file 'Data_1.h5', 
                  the output_name of the *.png files will be 'Data_1_ens1.png' and so on
    '''
    

    f = h5.File(path + '.h5', 'r')
    ensembles = f[dataset][...]
    f.close()
        
    all_neurons = range(ensembles.shape[1])
    n_neurons = ensembles.shape[1]
    n_ensembles = ensembles.shape[0]
    n_lag = ensembles.shape[2]
      
    all_neurons = []
    nonempty_ensembles = []
    neurons_color = []
    for i_ensemble in range(n_ensembles):

        neurons = range(ensembles.shape[1])
  
        if len(neurons) > 0:
            nonempty_ensembles.append(i_ensemble)

        for n in neurons:
            if n not in all_neurons:
                all_neurons.append(n)
                neurons_color.append(len(nonempty_ensembles) - 1)

    all_neurons = np.asarray(all_neurons)
    neurons_color = np.asarray(neurons_color)

    ensembles = ensembles[nonempty_ensembles, :, :]
    n_ensembles = ensembles.shape[0]

#    if sort:
#        permutation = np.argsort(all_neurons)
#        all_neurons = all_neurons[permutation]
#        neurons_color = neurons_color[permutation]

    mtx = np.zeros((n_neurons, n_ensembles, n_lag))

    for i_ensemble in range(n_ensembles):
        #mtx[:, i_ensemble, :] = ensembles[i_ensemble, :, :]
           mtx[:, i_ensemble, :] = ensembles[i_ensemble, all_neurons, :]

    vmin = np.min(mtx)
    vmax = np.max(mtx)

    #mtx[np.where(mtx > 0.05)] = 1
    #mtx[np.where(mtx > 0.2)] = 1
    
    if n_lag < 2:
        width = n_ensembles 
    else:
        width = n_ensembles * 8. * n_lag / 10.
    if n_neurons < 70:
        height = 10
    else:
        height = n_neurons / 10 * 1.5
    params = {
        'axes.labelsize': 50,
        'font.size': 50,
        'legend.fontsize': 50,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        "text.usetex": False,
        'figure.figsize': [width, height]
        }
        
    plt.rcParams.update(params)
    
    fig = plt.figure()
    gs = gridspec.GridSpec(1, n_ensembles)


    color_list = plt.cm.Set2(np.linspace(0, 1, n_ensembles))

    for i_ensemble in range(n_ensembles):
        ax = fig.add_subplot(gs[i_ensemble])

        if i_ensemble == 0:
            ax.set_ylabel("neuron")
        ax.set_xlabel("frame")
            
        cbar = ax.pcolor(mtx[:, i_ensemble, :], vmin=vmin, vmax=vmax, cmap=plt.cm.Blues)

        ax.set_ylim(0, mtx.shape[0])
        ax.set_xlim(0, mtx.shape[2])

        if n_neurons <= 20:
            if n_neurons <= 10:
                ax.set_yticks(np.arange(0, mtx.shape[0]+1, 1)+0.5, minor=False) 
            else:
                ax.set_yticks(np.arange(0, mtx.shape[0]+1, 5), minor=False)
        else:
            ax.set_yticks(np.arange(0, mtx.shape[0]+1, 10), minor=False)
#       
#       
#        ax.set_yticklabels(all_neurons)

#        active_neurons = np.unique(np.where(mtx[:, i_ensemble, :] > 0)[0])
#        for i, lbl in enumerate(ax.get_yticklabels()):
#            lbl.set_color(color_list[neurons_color[1]])              #anstatt 0 mal i gestanden
#            if i in active_neurons:
#                lbl.set_fontweight('heavy')
        if n_lag > 10:
            if n_lag > 20:
                if n_lag > 50:
                    ax.set_xticks(np.arange(0,mtx.shape[2],10) + 0.5)
                    ax.set_xticklabels(np.arange(0,mtx.shape[2],10) + 1)
                else:
                    ax.set_xticks(np.arange(0,mtx.shape[2],5) + 0.5)
                    ax.set_xticklabels(np.arange(0,mtx.shape[2],5) + 1)
            else:
                ax.set_xticks(np.arange(0,mtx.shape[2],2) + 0.5)
                ax.set_xticklabels(np.arange(0,mtx.shape[2],2) + 1)    
        else:
            ax.set_xticks(np.arange(mtx.shape[2]) + 0.5)
            ax.set_xticklabels(np.arange(mtx.shape[2]) + 1)

#        ax.set_yticks(np.arange(0, mtx.shape[0], 5), minor=True)
#        #ax.grid(True, linestyle='-', lw=2, which='minor', axis='y')

        ax.set_title("motif %d" % (nonempty_ensembles[i_ensemble] + 1))

    fig.colorbar(cbar)

    fig.savefig(path+naming+'.png', bbox_inches='tight')

    plt.close()

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='ensemble plotter')

    parser.add_argument('-o', '--output', default="", help="store plot as png")
    parser.add_argument('-dn', '--dataset', default="result/ensembles", help="dataset name (default: %(default)s)")
    parser.add_argument('-d', '--result', help="h5 file with the ensembles")

    args = parser.parse_args()
    
    plot_ensembles(args.result, args.dataset, args.output)
