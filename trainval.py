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

import os
import pandas as pd

from tqdm import tqdm
from src import utils as ut
from src import datasets
from src import losses
from scipy.io import savemat
from src.partition_rules import partition_rules
from src.selection_rules import VB_selection_rules
from src.selection_rules import FB_selection_rules
from src.update_rules import update_rules
from src import datasets
from src.partition_rules import partition_rules
from src.selection_rules import VB_selection_rules
from src.selection_rules import FB_selection_rules
from src.update_rules import update_rules

import argparse
from src import utils
from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader


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
    if not ut.is_valid_exp(exp_dict):
        return

    score_list_fname = os.path.join(savedir, 'score_list.pkl')
    if os.path.exists(score_list_fname):
        return hu.load_pkl(score_list_fname)

    score_list = train(dataset_name=exp_dict['dataset']['name'],
                       loss_name=exp_dict['dataset']['loss'],
                       block_size=exp_dict['block_size'],
                       partition_rule=exp_dict['partition'],
                       selection_rule=exp_dict['selection'],
                       update_rule=exp_dict['update'],
                       n_iters=exp_dict['max_iters'],
                       L1=exp_dict.get('l1', 0),
                       L2=0,
                       datasets_path=datadir)

    hu.save_pkl(score_list_fname, score_list)
    print('Experiment completed.')


def train(dataset_name, loss_name, block_size, partition_rule,
          selection_rule,
          update_rule, n_iters, L1, L2,  optimal=None,
          datasets_path=""):

    np.random.seed(1)
    # load dataset
    dataset = datasets.load(dataset_name, path=datasets_path)
    A, b, args = dataset["A"], dataset["b"], dataset["args"]

    args.update({"L2": L2, "L1": L1, "block_size": block_size,
                 "update_rule": update_rule})

    # loss function
    lossObject = losses.create_lossObject(loss_name, A, b, args)

    # Get partitions
    partition = partition_rules.get_partition(
        A, b, lossObject, block_size, p_rule=partition_rule)

    # Initialize x
    x = np.zeros(lossObject.n_params)

    score_list = []

    pbar = tqdm(desc="starting", total=n_iters, leave=True)

    ###### TRAINING STARTS HERE ############
    block = np.array([])
    for i in range(n_iters + 1):
        # Compute loss
        loss = lossObject.f_func(x, A, b)
        dis2opt = loss - \
            exp_configs.OPTIMAL_LOSS[dataset_name + "_" + loss_name]
        score_list += [{"loss": loss, "iteration": i, "selected": block}]

        stdout = ("%d - %s_%s_%s - dis2opt:%.16f - nz: %d/%d" %
                  (i, partition_rule, selection_rule, update_rule, dis2opt, (x != 0).sum(), x.size))
        print(stdout)

        # Check convergence
        if (i > 5 and (np.array_equal(work, np.where(x > 1e-16)[0]))):
            score_list[-1]["converged"] = dis2opt
        if (i > 5 and (dis2opt == 0 or dis2opt < 1e-8)):
            break

        # Check increase
        if (i > 0) and (loss > score_list[-1]["loss"] + 1e-6):
            raise ValueError("loss value has increased...")

        # Select block
        if partition is None:
            block, args = VB_selection_rules.select(
                selection_rule, x, A, b, lossObject, args, iteration=i)
        else:
            block, args = FB_selection_rules.select(
                selection_rule, x, A, b, lossObject, args, partition, iteration=i)

        # Update block
        x, args = update_rules.update(
            update_rule, x, A, b, lossObject, args=args, block=block, iteration=i)

    pbar.close()

    for score_dict in score_list:
        score_dict["loss"] -= exp_configs.OPTIMAL_LOSS[dataset_name + "_" + loss_name]

    return score_list


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
