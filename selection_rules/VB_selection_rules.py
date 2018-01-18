from . import tree_algorithms as ta

import numpy as np
from update_rules import update_rules as ur
#from pulp import *
import copy
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from itertools import cycle
########### ------------------------------ ADAPTIVE RULES
def select(rule, x, A, b, loss, args, iteration):
    """ Adaptive selection rules """
    n_params = x.size
    block_size = args["block_size"]
    it = iteration
    lipschitz = loss.lipschitz

    if "Tree" not in rule:
      assert block_size > 0
    else:
      assert block_size == -1
      
    g_func = loss.g_func

    
    if rule == "all":
       """ select all coordinates """
       block = np.arange(n_params)

    elif rule == "Random":
       """ randomly select a coordinate"""
       all_block = np.random.permutation(n_params)
       block = all_block[:block_size]

       #block = np.unravel_index(block,  (n_features, n_classes))


    elif rule in ["Perm", "Cyclic"]:
      """Select next coordinate"""
      
      if iteration % n_params == 0:
         args["perm_coors"] = np.random.permutation(n_params)

      emod = it % int((n_params/block_size))
      block = args["perm_coors"][emod*block_size: (emod + 1)*block_size]
      
      #block = np.unravel_index(block,  (n_features, n_classes))

    elif rule == "Lipschitz":
      """non-uniform sample based on lipschitz values"""
      L = lipschitz

      block = np.random.choice(x.size, block_size, replace=False,
                               p=L/L.sum())
    
    elif rule in ["GS"]:
      """ select coordinates based on largest gradients"""
      g = g_func(x, A, b, block=None)
      s = np.abs(g)

      block = np.argsort(s, axis=None)[-block_size:]

    elif rule in ["GSDLi", "GSD"]:
      """ select coordinates based on largest individual lipschitz"""
      L = lipschitz
      g = g_func(x, A, b, block=None)

      s = np.abs(g) / np.sqrt(L)
                                     
      block = np.argsort(s, axis=None)[-block_size:]

    elif rule in ["GSDHb"]:
      """ select coordinates based on the uper bound of the hessian"""
      g = g_func(x, A, b, block=None)

      if "GSD_L" not in args:
        Hb = loss.Hb_func(x, A, b, block=None)
        
        args["GSD_L"] = np.sum(np.abs(Hb), 1)

      s = np.abs(g) / np.sqrt(args["GSD_L"])
                                     
      block = np.argsort(s, axis=None)[-block_size:]

    elif rule in ["GSQ-IHT", "IHT"]:
      """ select coordinates based on largest individual lipschitz"""
      L = lipschitz
      if "Hb_IHT" not in args:
        args["Hb_IHT"] = loss.Hb_func(x, A, b, block=None)

        #args["mu_IHT"] = 1. / np.max(np.linalg.eigh(args["Hb_IHT"])[0])
        args["mu_IHT"] = 1. / largest_eigsh(args["Hb_IHT"], 1, which='LM')[0]
      Hb = args["Hb_IHT"]
      mu = args["mu_IHT"]

      G = g_func(x, A, b, block=None)

      d = G / np.sqrt(L)
      d_old = d.copy()

      for i in range(10):

        d = d - mu*(G + Hb.dot(d))
        ind = np.argsort(np.abs(d))
        d[ind[:-block_size]]= 0

        if np.linalg.norm(d_old - d) < 1e-10:

          block = ind[-block_size:]
          break
        #print "norm diff: %.3f" % np.linalg.norm(d_old - d)
        d_old = d.copy()
        block = ind[-block_size:]
      #block = np.where(d != 0)
      return np.array(block), args

    elif rule == "gsq-nn":
      """ select coordinates based on largest individual lipschitz"""
      g = g_func(x, A, b, block=None)
      L = lipschitz
      d = -g / L

      x_new = x + d
      neg = x_new < 0

      pos = (1 - neg).astype(bool)
      
      # SANITY CHECK
      assert x.size == (neg.sum() + pos.sum())

      s = np.zeros(x.size)
      d = -g[pos] / L[pos]
      s[pos] = g[pos] * d + (L[pos]/2.) * d**2

      d = - x[neg]
      s[neg] = g[neg] * d + (L[neg]/2.) * d**2
                        
      block = np.argsort(s, axis=None)[:block_size]
    
    elif rule in ["GSDTree", "GSTree","RTree", "GSLTree"]:

      """ select coordinates that form a forest based on BGS or BGSC """
      g_func = loss.g_func

      
      if "GSDTree" == rule:
        lipschitz = np.sum(np.abs(A), 1)
        score_list = np.abs(g_func(x, A, b, None)) / np.sqrt(lipschitz)
        sorted_indices = np.argsort(score_list)[::-1] 

      elif "GSLTree" == rule:
        lipschitz = lipschitz
        score_list = np.abs(g_func(x, A, b, None)) / np.sqrt(lipschitz)
        sorted_indices = np.argsort(score_list)[::-1]  

      elif "GSTree" == rule:
        score_list = np.abs(g_func(x, A, b, None))    
        sorted_indices = np.argsort(score_list)[::-1]  

      elif "RTree" == rule:
        sorted_indices = np.random.permutation(np.arange(A.shape[0]))

      block = ta.get_tree_slow(sorted_indices, adj=A)
      
      if iteration == 0:
        xr =  np.random.randn(*x.shape)
        xE, _ = ur.update("bpExact", xr.copy(), 
                                 A, b, loss, copy.deepcopy(args), block, iteration=iteration)
        xG, _ = ur.update("bpGabp", xr.copy(), 
                                A, b, loss, copy.deepcopy(args) , block, iteration=iteration)

        np.testing.assert_array_almost_equal(xE, xG, 3)

        print("Exact vs GaBP Test passed...")


    elif rule == "GSExactTree":
      """ select coordinates based on largest individual lipschitz"""

      g = g_func(x, A, b, block=None)

      s = np.abs(g)
      
      block_size = int(loss.n_params**(1./3))
      block = np.argsort(s, axis=None)[-block_size:]

    elif rule == "GSLExactTree":
      """ select coordinates based on largest individual lipschitz"""

      l = lipschitz
      g = g_func(x, A, b, block=None)

      s = np.abs(g) / np.sqrt(l)
      
      block_size = int(loss.n_params**(1./3))
      block = np.argsort(s, axis=None)[-block_size:]


    elif rule in ["TreePartitions", "RedBlackTree", 
                  "TreePartitionsRandom", 
                  "RedBlackTreeRandom"]:
      """ select coordinates that form a forest based on BGS or BGSC """
           
      g_func = loss.g_func 

      if "graph_blocks" not in args:       
        yb = args["data_y"]
        unlabeled = np.where(yb == 0)[0]
        Wb = args["data_W"][unlabeled][:, unlabeled]

        #################### GET GRAPH BLOCKS
        if args["data_lattice"] == False:     
          if rule == "RedBlackTree":
            graph_blocks = ta.get_rb_general_graph(Wb, L=lipschitz)

          elif rule == "TreePartitions":     
            graph_blocks = ta.get_tp_general_graph(Wb, L=lipschitz)

          elif rule == "RedBlackTreeRandom":
            graph_blocks = ta.get_rb_general_graph(Wb, L=np.ones(lipschitz.size))

          elif rule == "TreePartitionsRandom":     
            graph_blocks = ta.get_tp_general_graph(Wb, L=np.ones(lipschitz.size))


          else:

            raise ValueError("%s - No" % rule)

        if args["data_lattice"] == True:     
          if rule == "RedBlackTree":
            graph_blocks = ta.get_rb_indices(args["data_nrows"], 
                                             args["data_ncols"])

          elif rule == "TreePartitions":     
            graph_blocks = ta.get_tp_indices(args["data_nrows"], 
                                             args["data_ncols"])

          else:
            raise ValueError("%s - No" % rule)

          graph_blocks = ta.remove_labeled_nodes(graph_blocks, args["data_y"])

        
        #################### SANITY CHECK
        if rule in ["RedBlackTree", "RedBlackTreeRandom"]:
          # Assert all blocks have diagonal dependencies
          for tmp_block in graph_blocks:
            tmp = A[tmp_block][:, tmp_block]
            assert np.all(tmp == np.diag(np.diag(tmp)))

        elif rule in ["TreePartitions","TreePartitionsRandom"]:
          # Assert all blocks are forests/acyclic
          for tmp_block in graph_blocks:
            W_tmp = (Wb[tmp_block][:, tmp_block] != 0).astype(int)
            assert ta.isForest(W_tmp) 
        else:
          raise ValueError("%s - No" % rule)

        args["graph_blocks"] = cycle(graph_blocks)
       
        
      block = next(args["graph_blocks"])

      block.sort()  

      # check if block is diag
      # tmp = A[block][:, block]
      # assert np.all(tmp == np.diag(np.diag(tmp)))  

      if iteration == 0:
        x = np.random.randn(A.shape[1])
        xr =  np.random.randn(*x.shape)
        xE, _ = ur.update("bpExact", xr.copy(), 
                                 A, b, loss, copy.deepcopy(args), block, iteration)
        xG, _ = ur.update("bpGabp", xr.copy(), 
                                A, b, loss, copy.deepcopy(args) , block, iteration)

        np.testing.assert_array_almost_equal(xE, xG, 3)
        print("Exact vs GaBP Test passed...")

    else:
      raise ValueError("selection rule %s doesn't exist" % rule)

    
    if "Tree" not in rule:
      assert block_size == block.size


    assert np.unique(block).size == block.size
    block.sort()
    return block, args

