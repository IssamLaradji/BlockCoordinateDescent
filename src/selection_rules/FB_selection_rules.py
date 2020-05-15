
import numpy as np
from ..update_rules import update_rules as ur
#from pulp import *
import copy
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh


def select(rule, x, A, b, loss, args,  partition, iteration):
    fixed_blocks = partition
    
    if args["update_rule"] == "LA":
      if "LA_lipschitz" not in args:
        args["LA_lipschitz"] = np.ones(x.size)
      lipschitz = args["LA_lipschitz"]
      
    else:
      lipschitz = loss.lipschitz

    n_blocks = len(fixed_blocks)

    if rule == "Random":
      """ randomly select a block"""
      block = fixed_blocks[np.random.randint(0, n_blocks)]

      return block, args

    elif rule == "Cyclic":
      """ cycle across blocks"""
      block = fixed_blocks[iteration % len(fixed_blocks)]

      return block, args

    elif rule == "Lipschitz":
      """sample a random block proportional to L_b"""
      Lb_func = loss.Lb_func

      if "Lb_list" not in args:
        args["Lb_list"] = [] 
        n_blocks = len(fixed_blocks)

        # COMPUTE LB
        for block in fixed_blocks:
          Lb = Lb_func(x, A, b, block)

          #np.linalg.eig(A[:, features].T.dot(A[:, features]))[0]

          args["Lb_list"] += [Lb]

        args["Lb_list"] = np.array(args["Lb_list"])

      block = fixed_blocks[np.random.choice(n_blocks, p=args["Lb_list"]/args["Lb_list"].sum())]
      return block, args


    elif rule == "GS":
      """ select a block based on the largest gradient sum"""
      g_func = loss.g_func

      best_block = -1
      best_score = -1
      
      g = g_func(x, A, b, block=None)
      
      for i, block in enumerate(fixed_blocks):
        g_block = g[block]
        s = np.sum(np.abs(g_block))
        #print "bs: %d, func: %.3f" % (len(block), s)
        if s > best_score:
          best_score = s
          best_block = i
          
      
      return fixed_blocks[best_block], args



    elif rule == "GSL":
      """ select a block based on the maximum eigen value"""
      g_func = loss.g_func
      Lb_func = loss.Lb_func
      best_block = -1
      best_score = -1


      g = g_func(x, A, b, block=None)

      if args["update_rule"] == "LA":

          args["Lb_list"] = [] 
          n_blocks = len(fixed_blocks)

          # COMPUTE LB
          for block in fixed_blocks:
            Lb = lipschitz[block]
            #np.linalg.eig(A[:, features].T.dot(A[:, features]))[0]

            args["Lb_list"] += [Lb]

          args["Lb_list"] = np.array(args["Lb_list"])

      else:
        if "Lb_list" not in args:
          args["Lb_list"] = [] 
          n_blocks = len(fixed_blocks)

          # COMPUTE LB
          for block in fixed_blocks:
            Lb = Lb_func(x, A, b, block)

            #np.linalg.eig(A[:, features].T.dot(A[:, features]))[0]

            args["Lb_list"] += [Lb]

          args["Lb_list"] = np.array(args["Lb_list"])


      for i, block in enumerate(fixed_blocks):
        g_block = g[block]
        L_block = args["Lb_list"][i]

        step_size = 1. / np.sqrt(L_block)

        s = np.sum(np.abs(g_block) * step_size) 

        if s > best_score:
          best_score = s
          best_block = i
       
      return fixed_blocks[best_block], args


    elif rule in ["GSD"]:
      """ select a block based on the individual Lis"""
      g_func = loss.g_func

      best_block = -1
      best_score = -1
      
      g = g_func(x, A, b, block=None)
      l = lipschitz

      for i, block in enumerate(fixed_blocks):
        g_block = g[block]
        l_block = l[block]
        

        s = np.sum(np.abs(g_block) / np.sqrt(l_block)) 

        if s > best_score:
          best_score = s
          best_block = i
       
      return fixed_blocks[best_block], args

    elif rule == "gsq-nn":
      """ select coordinates based on largest individual lipschitz"""
      g_func = loss.g_func

      best_block = -1
      best_score = -1
      
      g = g_func(x, A, b, block=None)
      l = lipschitz

      for i, block in enumerate(fixed_blocks):

        Gb = g[block]
        Lb = l[block]

        xb = x[block]
        d = -Gb / Lb

        x_new = xb + d
        neg = x_new < 0

        pos = (1 - neg).astype(bool)

        assert xb.size == (neg.sum() + pos.sum())

        s = np.zeros(xb.size)
        d = -Gb[pos] / Lb[pos]
        s[pos] = np.abs(Gb[pos] * d + (Lb[pos]/2.) * d**2)

        d = - xb[neg]
        s[neg] = np.abs(Gb[neg] * d + (Lb[neg]/2.) * d**2)


        s = np.sum(s) 
        
        if s > best_score:
          best_score = s
          best_block = i
       
      return fixed_blocks[best_block], args


    elif rule == "GSQ":
      """ select a block based on the hessian"""
      g_func = loss.g_func
      Hb_func = loss.Hb_func

      best_block = -1
      best_score = -1
        
      g = g_func(x, A, b, block=None)
      if "GSQ_Hb" not in args:
        args["GSQ_Hb"] = Hb_func(x, A, b, block=None)

      Hb = args["GSQ_Hb"]

      #H = h_func(x, A, b, args, None)
      if "GSQ_Hb_inv" not in args:
       args["GSQ_Hb_inv"] = [0]*len(fixed_blocks)

      for i, block in enumerate(fixed_blocks):
        g_block = g[block]
        
        if isinstance(args["GSQ_Hb_inv"][i], (int,float)):
          hb =  Hb[block][:,block]
          args["GSQ_Hb_inv"][i] = np.linalg.inv(hb + 1e-10*np.eye(len(hb)))

        s = np.dot(args["GSQ_Hb_inv"][i], g_block)
        s = np.dot(g_block.T, s)
        s = np.sqrt(s)

        if s > best_score:
          best_score = s
          best_block = i
       
      return fixed_blocks[best_block], args
      
    else:
      raise ValueError("selection rule %s doesn't exist" % rule)