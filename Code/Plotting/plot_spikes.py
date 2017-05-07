import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import argparse
import sys, os
import json

patterns = 'aabacbddccdacadbbaacbcdb'
flavor = '121112122211122212121112'
starts = range(31, 1000, 21+20)

def plot_spikes(data, title, ylabel):
    f = plt.figure()
    a = f.add_subplot(111)

    f.set_size_inches(22, 11)

    for i_neuron in range(data.shape[0]):
        spikes = data[i_neuron, :]
        a.vlines(np.where(spikes > 0)[0], i_neuron, i_neuron + 1.0, color='k')
    a.set_ylim(0, data.shape[0])

    if title:
        a.set_title('%s' % title)
    a.set_xlabel('frame')
    a.set_ylabel(ylabel)

    a.set_ylim(0, data.shape[0])
    ystart = 0
    if data.shape[0] < 20:
        a.set_yticks(np.arange(ystart, data.shape[0] + ystart), minor=False)
    else:
        skip = 20
        if data.shape[0] > 200:
            skip = 100
        a.set_yticks(np.arange(ystart, data.shape[0] + ystart + skip, skip), minor=False)

    if data.shape[0] < 200:
        a.set_yticks(np.arange(ystart, data.shape[0] + ystart), minor=True)

    a.grid(True, axis='y', linestyle='-', which='minor')
    a.set_xlim(0, data.shape[1])
    a.set_ylim(0, data.shape[0])

    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spike plotter')
    
    parser.add_argument('-d','--data',help="name of the .h5 file containing the data to be plotted, without .h5")
    parser.add_argument('-o', '--output', help="store plot as png")
    parser.add_argument('-q', '--quiet', help="don't show plot", action='store_true')
    parser.add_argument('-dn', '--dataset', default="result/spikes", help="dataset name (default: %(default)s)")
    parser.add_argument('-t', '--title', help="plot title")
    parser.add_argument('-m', '--min-threshold', type=float, help="minimum spike threshold", default=0)
    parser.add_argument('-swap', '--swap-axis', action='store_true')
    parser.add_argument('-y', '--y-label', default="assembly")
    parser.add_argument('--y-start', type=int, default=1)


    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--shade', action='append', help="add vertical shades. format: start,end,color id")
    group.add_argument('-l', '--load-shade', help="load vertical shades from json file")

    
    args = parser.parse_args()

    f = h5.File(args.data+'.h5', 'r')
    spikes = f[args.dataset][...]
    f.close()

    if args.swap_axis:
        spikes = spikes.swapaxes(0, 1)

    spikes[np.where(spikes < args.min_threshold)] = 0

    
    f = plot_spikes(spikes, args.title, args.y_label)

    #if not args.quiet:
    #    f.show()
    #    input("press any button to exit")

    if args.output:
        f.set_size_inches(30,15)
        f.savefig(args.output, bbox_inches="tight")
