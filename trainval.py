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
import train
import numpy as np

from src import datasets


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

    # BCD train
    # ==================
    # Ignore the following combinations
    if ((exp_dict['partition'] != "VB" and exp_dict['selection'] == "BGSC") or
        (exp_dict['partition'] != "VB" and exp_dict['selection'] == "OMP") or
        (exp_dict['partition'] == "VB" and exp_dict['selection'] == "GSQ") or
        (exp_dict['partition'] != "VB" and "GSQ-" in exp_dict['selection']) or
        (exp_dict['partition'] != "VB" and exp_dict['selection'] == "GSC") or
        (exp_dict['partition'] != "VB" and exp_dict['selection'] == "Perm") or
        (exp_dict['partition'] == "VB" and exp_dict['selection'] == "BGSL") or
        (exp_dict['partition'] == "VB" and exp_dict['selection'] == "GSL") or
        (exp_dict['partition'] == "VB" and exp_dict['selection'] == "cCyclic")or
        (exp_dict['partition'] != "VB" and exp_dict['selection'] == "IHT")or
        (exp_dict['partition'] != "VB" and exp_dict['selection'] == "GSDHb")):
        print('Experiment will not run...')
        return

    score_list = train.train(dataset_name=exp_dict['dataset']['name'],
                            loss_name=exp_dict['dataset']['loss'],
                            block_size=exp_dict['block_size'],
                            partition_rule=exp_dict['partition'],
                            selection_rule=exp_dict['selection'],
                            update_rule=exp_dict['update'],
                            n_iters=exp_dict['max_iters'],
                            L1=exp_dict.get('l1',0),
                            L2=0,
                            datasets_path=datadir)

    hu.save_pkl(os.path.join(savedir, 'score_list.pkl'), score_list)
    print('Experiment completed.')

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
