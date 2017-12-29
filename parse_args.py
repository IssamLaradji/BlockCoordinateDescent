import argparse
import os
from base import utils as ut
import numpy as np 
import sys 
import json

def parse():
    io_args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-do','--data_obj',nargs='+', default =["exp2-logistic"])  
    parser.add_argument('-pn','--plot_names',nargs='+', type=str, default=["None:None"])  
    parser.add_argument('-r2c','--run2conv', type=int,  default =0)  
    parser.add_argument('-pc','--plot_colors', nargs='+', type=str)  
    parser.add_argument('-d','--dataset_names',nargs='+',  default =["exp2"])  
    parser.add_argument('-l','--loss_names',nargs='+', default =["logistic"])
    parser.add_argument('-s','--s_rules', nargs='+', default =["random"])
    parser.add_argument('-u','--u_rules',nargs='+', default=["quadraticLi"])
    parser.add_argument('-scale','--scale', type=int, default=0)
    parser.add_argument('-br','--breakpoint', type=int, default=0)
    parser.add_argument('-b','--blockList', type=int, nargs='+', default=[5])
    parser.add_argument('-n','--n_iters', default=10, type=int)
    parser.add_argument('-p','--p_rules', nargs='+', default=["Ada"])
    parser.add_argument('-e','--exp', nargs='+', default=[None])
    parser.add_argument('-desc','--description', default="default")
    parser.add_argument('-f','--show_fig', type=int, default=0)
    parser.add_argument('-r','--reset',  nargs='+', type=str, default=[""])   
    parser.add_argument('-itp','--iterPlot',  type=int, default=None)  
    parser.add_argument('-psu','--partition_selection_update', nargs='+', default=[None])
    parser.add_argument('-l1','--L1', type=float, default=0)
    parser.add_argument('-l2','--L2', type=float, default=0)
    parser.add_argument('-t','--test_grad', type=int, default=0)
    parser.add_argument('-ti','--timeit', type=int, default=0)
    parser.add_argument('-ad','--assert_decrease', type=int, default=1)
    parser.add_argument('-d2o','--distance_to_optimal', type=int, default=0)
    parser.add_argument('-L_approx','--L_approx', type=int, default=0)
    parser.add_argument('-format','--format', default="pdf")
    parser.add_argument('-v','--verbose', default="verbose")
    parser.add_argument('-yloss','--yloss', default="loss", choices=["loss", "log"])
    parser.add_argument('-ml','--minLoss', type=float, default=None)
    parser.add_argument('-title','--add_title', type=int, default=1)
    parser.add_argument('-ylim','--ylim', nargs='+', default=[None])
    parser.add_argument('-ylimIgnore','--ylimIgnore', nargs='+', 
                        default=[None])
    parser.add_argument('-in','--init',
                        default="zeros")

    io_args = parser.parse_args()

    if io_args.exp[0] != None:
        argsList = []
        for exp in io_args.exp:
            
            exp_args = ut.parseArg_json(exp, parser, fname="exps.json")
            
            ### OVERRIDE WITH IO_ARGS
            for key in ["assert_decrease", "test_grad", "reset"]:
                vars(exp_args)[key] =  vars(io_args)[key]

            vars(exp_args)["expName"] = exp
            argsList += [exp_args]
    else:
        argsList = [io_args]


    return argsList
