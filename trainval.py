from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets
from src import utils as ut


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

cudnn.benchmark = True


def trainval(exp_dict, savedir_base, datadir, reset=False, num_workers=0):
    # bookkeepting stuff
    # ==================
    pprint.pprint(exp_dict)
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    if reset:
        hc.delete_and_backup_experiment(savedir)

    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    print("Experiment saved in %s" % savedir)

    # Dataset
    # ==================
    # train set
    plots_path = ROOT + "/Checkpoints/CoordinateDescent/Figures/"
    logs_path = ROOT + "/Checkpoints/CoordinateDescent/Logs/"
    datasets_path = ROOT + "/Datasets/CoordinateDescent/"

    # LOOP OVER EXPERIMENTS
    for args in argsList:   
        plotList = []
        historyDict = {}

        ######## TRAIN STAGE #########

        # Loop over datasets
        for dataset_name, loss_name in zip(args.dataset_names, 
                                           args.loss_names):
            figureList = []
            # Loop over loss names
            for block_size in args.blockList:  

                traceList = []
                # Loop over p, s, and u rules
                for p, s, u in product(args.p_rules, args.s_rules,
                                       args.u_rules):

                    # Ignore the following combinations
                    if ((p != "VB" and s == "BGSC") or
                        (p != "VB" and s == "OMP") or
                        (p == "VB" and s == "GSQ") or
                        (p != "VB" and "GSQ-" in s) or
                        (p != "VB" and s == "GSC") or
                        (p != "VB" and s == "Perm") or
                        (p == "VB" and s == "BGSL") or
                        (p == "VB" and s == "GSL") or
                        (p == "VB" and s == "cCyclic")or
                        (p != "VB" and s == "IHT")or
                        (p != "VB" and s == "GSDHb")):
                        continue

                    history = train.train(dataset_name=dataset_name,
                                          loss_name=loss_name,
                                          block_size=block_size,
                                          partition_rule=p,
                                          selection_rule=s,
                                          update_rule=u,
                                          n_iters=args.n_iters,
                                          reset=args.reset,
                                          L1=args.L1,
                                          L2=0,
                                          root=ROOT,
                                          logs_path=logs_path,
                                          datasets_path=datasets_path)

                    legend = ut.legendFunc(p, s, u, args.p_rules, args.s_rules, 
                                           args.u_rules, args.plot_names)

                    

                    if "converged" in history.columns:
                      ind = np.where(np.isnan(np.array(history["converged"])))[0][-1] + 1
                      converged = {"Y":history["converged"][ind],
                                   "X":ind}
                    else:
                      converged = None

                    traceList += [{"Y":np.array(history["loss"]), 
                                   "X":np.array(history["iteration"]),
                                   "legend":legend,
                                   "converged":converged}]

                    historyDict[legend] = history

                if block_size == -1:
                  xlabel = "Iterations"
                else:
                  xlabel = "Iterations with %d-sized blocks" % block_size
                  
                figureList += [{"traceList":traceList,
                                "xlabel":xlabel,
                                "ylabel":("$f(x) - f^*$ for %s on Dataset %s" % 
                                         (loss2name[loss_name], dataset_name.upper())),
                                "yscale":"log"}]
            
            plotList += [figureList]
            


        ########## PLOT STAGE #########
        fig = plot.plot(plotList, expName=args.expName, path=plots_path)

        ut.visplot(fig, win=args.expName)
        matplotlib.pyplot.close()




        ########## SAVE EXP HISTORY ##########
        ut.save_pkl(logs_path + args.expName + ".pkl", historyDict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)

    args = parser.parse_args()

    # Collect experiments
    # ===================
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Run experiments
    # ===============
    if args.run_jobs:
        import usr_configs as uc
        uc.run_jobs(exp_list, args.savedir_base, args.datadir)
    
    else:
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    datadir=args.datadir,
                    reset=args.reset,
                    num_workers=args.num_workers)
