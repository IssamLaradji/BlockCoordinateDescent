import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.16f}".format(x)})

import os
import pandas as pd

from tqdm import tqdm

from src import datasets
from src import losses
from scipy.io import savemat
from src.partition_rules import partition_rules
from src.selection_rules import VB_selection_rules
from src.selection_rules import FB_selection_rules
from src.update_rules import update_rules
from src.base import utils as ut



OPTIMAL_LOSS = {"A_ls": 8.1234048724830014e-25,
                "A_lsl1nn":6725753.5240152273327112,
                "B_lg": 5.0381920857462139e-15,
                "C_sf": 1.0881194612011313e-11, 
                "D_bp": -1045999575.2270696, 
                "E_bp": -717.708822011346}

work = np.array([84,  220,  478,  558,  596,  753, 1103, 2009, 2044, 2301, 2410,
       2514, 2746, 3694, 4054, 4249, 4429, 4764, 5110, 5299, 5340, 5447,
       5680, 5899, 6254, 6256, 6412, 6518, 6538, 6587, 6770, 6796, 6848,
       6881, 6917, 6975, 7055, 7121, 7188, 7456, 8217, 8479, 8925, 9190,
       9583, 9681, 9690, 9692, 9793, 9811, 9992])

def train(dataset_name, loss_name, block_size, partition_rule, 
          selection_rule, 
          update_rule, n_iters, L1, L2,  optimal=None, 
          datasets_path=""):
    
    np.random.seed(1)
    # load dataset
    dataset = datasets.load(dataset_name, path=datasets_path)
    A, b, args = dataset["A"], dataset["b"], dataset["args"]
    
    args.update({"L2":L2, "L1":L1, "block_size":block_size, 
                    "update_rule":update_rule})

    # loss function
    lossObject = losses.create_lossObject(loss_name, A, b, args)

    # Get partitions
    partition = partition_rules.get_partition(A, b, lossObject, block_size, p_rule=partition_rule)

    # Initialize x
    x = np.zeros(lossObject.n_params)

    score_list = []

    pbar = tqdm(desc="starting", total=n_iters, leave=True)

    ###### TRAINING STARTS HERE ############
    block = np.array([])
    for i in range(n_iters + 1):
        # Compute loss
        loss = lossObject.f_func(x, A, b)
        dis2opt = loss - OPTIMAL_LOSS[dataset_name + "_" + loss_name]
        score_list += [{"loss":loss, "iteration":i, "selected":block}]


        # if i == 10:
        #     import ipdb; ipdb.set_trace()  # breakpoint c7301fd5 //

        stdout = ("%d - %s_%s_%s - dis2opt:%.16f - nz: %d/%d" % 
                    (i, partition_rule, selection_rule, update_rule, dis2opt, (x!=0).sum(), x.size) )   
        #pbar.set_description(stdout)
        print(stdout)

        # # Check convergence
        if (i > 5 and (np.array_equal(work, np.where(x>1e-16)[0]))):
            score_list[-1]["converged"] = dis2opt

        if (i > 5 and (dis2opt == 0 or dis2opt < 1e-8)):
            break


        # Check increase
        if (i > 0) and (loss > score_list[-1]["loss"] + 1e-6): 
            raise ValueError("loss value has increased...")

        # Select block
        if partition is None:
            block, args = VB_selection_rules.select(selection_rule, x, A, b, lossObject, args, iteration=i)

        else:
            block, args = FB_selection_rules.select(selection_rule, x, A, b, lossObject, args, partition, iteration=i)

        # Update block
        x, args = update_rules.update(update_rule, x, A, b, lossObject, args=args, block=block, iteration=i)

    pbar.close()

    for score_dict in score_list:
        score_dict["loss"] -= OPTIMAL_LOSS[dataset_name + "_" + loss_name]
    
    return score_list


