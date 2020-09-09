
import numpy as np 

import json, shlex
import logging
import sys
from shutil import copyfile
import os
import glob
import networkx as nx
import matplotlib.pyplot as plt

import sys 
sys.path.append('..')
from loaders import LabelProp
from algorithms import tree_algorithms as ta

def show_lattice(selected=None, nrows=10, ncols=10, ratio=8.,name="img"):
    if selected is None:
        selected = np.zeros(nrows *  ncols)  
    #selected[:15] = 1
    # DONE FOR ORDERING
    # selected = np.reshape(selected, (nrows, ncols))
    # selected = np.fliplr(selected)
    # selected = selected.ravel()


    G=nx.grid_2d_graph(nrows, ncols)

    labels = dict( ((i,j), i + (ncols-1-j) * nrows ) for i, j in G.nodes() )

    # labels = np.arange(nrows*ncols)

    colors = np.zeros(len(labels), dtype=object)


    colors[:] = "w"


    
    #colors[selected == 1] = "r"

    nx.relabel_nodes(G,labels,False)
    inds=list(labels.keys())
    vals=list(labels.values())
    inds=[(nrows-j-1,ncols-i-1) for i,j in inds]
    pos=dict(list(zip(vals,inds)))



    for c, node in enumerate(G):
        if selected[node] == -1:
            colors[c] = "r"

        if selected[node] == 1:
            colors[c] = "k"
        if selected[node] == 2:
            colors[c] = "y"

    nodes = nx.draw_networkx_nodes(G, pos, with_labels=False, node_color=list(colors), node_size = 250*(ratio/nrows))
    # Set edge color to red
    nodes.set_edgecolor('k')
    nx.draw_networkx_edges(G, pos)

    # nx.draw_networkx(G, pos=pos, with_labels=False, edge_color="k", node_size = 250, 
    #                  node_color=list(colors))
 
    plt.axis('off')
    #plt.savefig("%s.png" % name, bbox_inches='tight')
    #plt.savefig("%s.eps" % name, bbox_inches='tight')
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("%s.pdf" % name, bbox_inches = 'tight',
    pad_inches = 0)
    #plt.show()


import optparse


# python lattice.py -r 14 -c 14 -s 5 -m greedy -n GTree_small
# python lattice.py -r 14 -c 14 -s 5 -m rb -n RedBlack_small
# python lattice.py -r 14 -c 14 -s 5 -m tp -n treePartition_small

if __name__ == "__main__":
    parser = optparse.OptionParser()

    parser.add_option('-r', '--nrows',
        action="store", dest="nrows")
    parser.add_option('-c', '--ncols',
        action="store", dest="ncols")
    parser.add_option('-s', '--size',
        action="store", dest="size")
    parser.add_option('-m', '--mode',
        action="store", dest="mode")
    parser.add_option('-n', '--name',
        action="store", dest="name")
    options, args = parser.parse_args()
    nrows, ncols = int(options.nrows), int(options.ncols)
    np.random.seed(1)
    if options.mode == "rb":

        red, black = LabelProp.getRBIndices(nrows, ncols)
        selected = np.zeros(nrows*ncols)
        selected[red] = 1
        selected[black] = -1

    elif options.mode == "tp":
        
        white, black = LabelProp.getPTreeIndices(nrows, ncols)
        selected = np.zeros(nrows*ncols)

        selected[white] = 1
        selected[black] = 3

    elif options.mode == "greedy":
        adj, _ = LabelProp.ising1(nrows=nrows, ncols=nrows)
        adj[adj!=0] = 1
        
        sorted_indices = np.random.permutation(np.arange(adj.shape[0]))
        block = ta.get_tree_slow(sorted_indices, adj=adj)
        selected = np.zeros(nrows*ncols)

        selected[block] = 1
        


    show_lattice(selected=selected, nrows=nrows, ncols=ncols, 
            ratio=float(options.size), name=options.name)

    # nrows, ncols = 15, 15
    # red, black = LabelProp.getRBIndices(nrows, ncols)
    # selected = np.zeros(nrows*ncols)
    # selected[red] = 1
    # selected[black] = -1
    # show_lattice(selected=selected, nrows=nrows, ncols=ncols, name="img")
