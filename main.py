import matplotlib
matplotlib.use('agg')

import numpy as np
import train
import parse_args

from itertools import product
from base import utils as ut
from base import plot

ROOT = "/mnt/AIDATA/home/issam.laradji"

loss2name = {"ls": "Least Squares", "lg":"Logistic", 
             "sf":"Softmax", "bp":"Quadratic",
             "lsl1nn":"Non-negative Least Squares"}

if __name__ == "__main__":
    argsList = parse_args.parse()

    # DEFINTE PATHs
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