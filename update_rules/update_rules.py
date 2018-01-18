# -*- coding: utf-8 -*-
import numpy as np
from . import line_search
import cvxopt

import utils as ut 
from cvxopt import solvers, matrix, spmatrix
from scipy import sparse
from cvxopt import matrix, solvers
from scipy.optimize import minimize

cvxopt.solvers.options['show_progress'] = False

def update(rule, x, A, b, loss, args, block, iteration):
  f_func = loss.f_func
  g_func = loss.g_func
  h_func = loss.h_func
  lipschitz = loss.lipschitz
  #mipschitz = loss.mipschitz

  # L2 = args["L2"]
  
  block_size = block.size
  param_size = x.size

  if rule in ["quadraticEg", "Lb"]:    
    """This computes the eigen values of the lipschitz values corresponding to the block"""
    # WE NEED TO DOUBLE CHECK THIS!
    G = g_func(x, A, b, block)
    L_block =loss.Lb_func(x, A, b, block)
    x[block] = x[block] - G / L_block

    return x, args

  elif rule in ["newtonUpperBound", "Hb"]:    

    G = g_func(x, A, b, block)
    H = loss.Hb_func(x, A, b, block)
    d = - np.linalg.pinv(H).dot(G)

    x[block] = x[block] + d

    return x, args

  elif rule == "LA":
    G = g_func(x, A, b, block)
    L_block =loss.Lb_func(x, A, b, block)
    
    Lb = np.max(args["LA_lipschitz"][block])
    
    while True:
      x_new = x.copy()
      x_new[block] = x_new[block] - G / Lb

      RHS = f_func(x,A,b) - (1./(2. * Lb)) * (G**2).sum()
      LHS = f_func(x_new,A,b)
      
      if LHS <= RHS:
        break

      Lb *= 2.

    args["LA_lipschitz"][block] = Lb

    return x_new, args

  # Line Search
  elif rule in ["LS"]:    

    H = h_func(x, A, b, block)

    g = g_func(x, A, b, block)

    f_simple = lambda x: f_func(x, A, b)
    d_func = lambda alpha: (- alpha * np.dot(np.linalg.pinv(H), g))

    

    alpha = line_search.perform_line_search(x.copy(), g, 
                                block, f_simple, d_func, alpha0=1.0,
                                proj=None)

    x[block] = x[block] + d_func(alpha)
    return x, args

  ### Constrained update rules
  elif rule in ["Lb-NN"]:
    G = g_func(x, A, b, block)
    L_block =loss.Lb_func(x, A, b, block)
    x[block] = x[block] - G / L_block

    x[block] = np.maximum(x[block], 0.)

    return x, args

  elif rule == "TMP-NN":
    L = lipschitz[block]

    grad_list = g_func(x, A, b, block)
    hess_list = h_func(x, A, b, block)

    H = np.zeros((block_size, block_size))
    G = np.zeros(block_size) 

    # The active set is on the bound close to x=0
    active = np.logical_and(x[block] < 1e-4, grad_list > 0)
    work = np.logical_not(active)

    # active
    ai = np.where(active == 1)[0]
    gA = grad_list[active]

    G[ai] = gA / (np.sum(L[active]))
    H[np.ix_(ai, ai)] = np.eye(ai.size)
    # work set
    wi = np.where(work == 1)[0]

    gW = grad_list[work]
    hW = hess_list[work][:, work]

    G[wi] = gW
    H[np.ix_(wi, wi)] = hW

    # Perform Line search
    alpha = 1.0
    
    u_func = lambda alpha: (- alpha * np.dot(np.linalg.inv(H), G))
    f_simple = lambda x: f_func(x, A, b, assert_nn=0)

    alpha = line_search.perform_line_search(x.copy(), G, 
                              block, f_simple, u_func, alpha0=1.0,
                                proj=lambda x: np.maximum(0, x))


    x[block] = np.maximum(x[block] + u_func(alpha), 0)

    return x, args

  elif rule == "qp-nn":
    cvxopt.setseed(1)
    non_block = np.delete(np.arange(param_size), block)
    k = block.size

    # 0.5*xb ^T (Ab^T Ab) xb + xb^T[Ab^T (Ac xc - b) + lambda*ones(nb)]
    Ab = matrix(A[:, block])
    bb = matrix(A[:, non_block].dot(x[non_block]) - b)

    P = Ab.T*Ab
    q = (Ab.T*bb + args["L1"]*matrix(np.ones(k)))

    G = matrix(-np.eye(k))
    h = matrix(np.zeros(k))
    x_block = np.array(solvers.qp(P=P, q=q, 
                                G=G, h=h, solver = "glpk")['x']).ravel()

    # cvxopt.solvers.options['maxiters'] = 1000
    cvxopt.solvers.options['abstol'] = 1e-16
    cvxopt.solvers.options['reltol'] = 1e-16
    cvxopt.solvers.options['feastol'] = 1e-16
    x[block] = np.maximum(x_block, 0)


    return x, args

  ### BELIEF PROPAGATION ALGORITHMS

  elif rule == "bpExact":
      n_params = x.size
      all_indices = np.arange(n_params)
      
      non_block_indices = np.delete(all_indices, block)
      
      A_bc = A[block][:, non_block_indices]
      A_bb = A[block][:, block]
      
      b_prime = A_bc.dot(x[non_block_indices]) - b[block]

      x[block] = np.linalg.inv(A_bb).dot(- b_prime)  # are you missing the x[block] + ?
      # Ans:
      # No, this is the exact update of the objective function formulation under
      # Appendix B. Derivation of Block Belief Propagation Update.

      return x, args


  elif rule == "bpGabp":
      A_sub = A[block][:, block]

      ######## ADDED
      _, n_features = A.shape
      all_indices = np.arange(n_features)
      non_block_indices = np.delete(all_indices, block)
      
      A_bc = A[block][:, non_block_indices]
      b_sub = A_bc.dot(x[non_block_indices]) - b[block]
      b_sub = - b_sub
      
      #########
      max_iter = 100
      epsilon = 1e-8
      
      #import pdb; pdb.set_trace()
      P = np.diag(np.diag(A_sub))
      U = np.diag(b_sub / np.diag(A_sub))
      
      n_features = A_sub.shape[0]
      
      # Stage 2 - iterate
      for iteration in range(max_iter):
         # record last round messages for convergence detection
         old_U = U.copy(); 
      
         for i in range(n_features):
             for j in range(n_features):
               
               if (i != j and A_sub[i,j] != 0):
                   # Compute P i\j - line 2
                   p_i_minus_j = np.sum(P[:,i]) - P[j,i]  
                   assert(p_i_minus_j != 0);
              
                   # Compute P ij - line 2
                   P[i,j] = -A_sub[i,j] * A_sub[j,i] / p_i_minus_j;
                   
                   # Compute U i\j - line 2
                   h_i_minus_j = (np.sum(P[:,i] * U[:,i]) - P[j,i]*U[j,i]) / p_i_minus_j;

                   # Compute U ij - line 3
                   U[i,j] = - A_sub[i,j] * h_i_minus_j / P[i,j];
                   #import pdb;pdb.set_trace()
                   
         
         # Stage 3 - convergence detection
         if (np.sum(np.sum((U - old_U)**2)) < epsilon):
               #print 'GABP converged in round %d ' % iteration
               break
       
      # Stage 4 - infer
      Pf = np.zeros(n_features);
      x_tmp = np.zeros(n_features);
      
      for i in range(n_features):
         Pf[i] = np.sum(P[:,i]); 
         x_tmp[i] = np.sum(U[:,i] * P[:,i]) / Pf[i];

      ##### Exact
      #x_exact = block_update(A, b, theta, block)
      x[block] = x_tmp

      
      return x, args

  else:
    print(("update rule %s doesn't exist" % rule))
    raise
