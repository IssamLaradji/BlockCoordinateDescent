import sys
from haven import haven_jupyter as hj
from haven import haven_results as hr
import pandas as pd
import os
import numpy as np
import pylab as plt
from haven import haven_utils as hu
from src import utils as ut
import argparse, exp_configs
from src import pretty_plot


def filter_exp_list(exp_list):
    # Ignore the following combinations
    exp_list_new = []
    for exp_dict in exp_list:
        if not ut.is_valid_exp(exp_dict):
            continue
        exp_list_new += [exp_dict]
        
    return exp_list_new

def get_one_plot(exp_list, savedir_base, plot_names=None):
    traceList = []
    p_rules=np.unique([e['partition'] for e in exp_list])
    s_rules =np.unique([e['selection'] for e in exp_list])
    u_rules = np.unique([e['update'] for e in exp_list])
        
    score_list_list = hr.get_score_lists(exp_list, savedir_base)
    assert(len(exp_list)== len(score_list_list))
    for exp_dict, score_list in zip(exp_list, score_list_list):
        # single figure
        score_df = pd.DataFrame(score_list)
        
        if "converged" in score_df.columns:
          ind = np.where(np.isnan(np.array(score_df["converged"])))[0][-1] + 1
          converged = {"Y":score_df["converged"][ind],
                       "X":ind}
        else:
          converged = None
    
        legend = ut.legendFunc(exp_dict['partition'], exp_dict['selection'], exp_dict['update'], 
                               p_rules, s_rules, u_rules, plot_names=plot_names)
        trace = {"Y":np.array(score_df["loss"]), 
                                   "X":np.array(score_df["iteration"]),
                                   "legend":legend,
                                   "converged":converged}
        traceList += [trace]
    return traceList

def get_dataset_plots(exp_list, plot_names, savedir_base):
    figureList = []
    
    loss_name = exp_list[0]['dataset']['loss']
    dataset = exp_list[0]['dataset']['name'].upper()
    xlabel = 'Iterations'
    
    exp_list_list = hr.group_exp_list(exp_list, groupby_list=['block_size'])
    exp_list_list.sort(key=lambda x:x[0]['block_size'])
    # across blocks
    for exp_list_bs in exp_list_list:
        block_size = exp_list_bs[0]['block_size']
        if block_size == -1:
            xlabel = "Iterations"
        else:
            xlabel = "Iterations with |b|=%d" % block_size
                
        trace_list = get_one_plot(exp_list_bs, savedir_base, plot_names=plot_names)
        figureList +=  [{'traceList':trace_list,
                          "xlabel":xlabel,
                          "ylabel":("$f(x) - f^*$ for %s on Dataset %s" % 
                                         (loss_name, dataset)),
                          "yscale":"log"}]
    return figureList


def plot_exp_list(exp_list, savedir_base, exp_name, outdir='figures', plot_names=None):
    exp_list = filter_exp_list(exp_list)
    exp_list_list = hr.group_exp_list(exp_list, groupby_list=['dataset'])
    plotList = []

    # across dataset
    for exp_list_dataset in exp_list_list:
        plotList += [get_dataset_plots(exp_list_dataset, plot_names, savedir_base)]

    nrows = len(plotList)
    ncols = len(plotList[0])

    # Main plot
    pp_main = pretty_plot.PrettyPlot(title=exp_name, 
                                axFontSize=14,
                                axTickSize=11,
                                legend_size=8,
                                figsize=(5*ncols,4*nrows),
                                legend_type="line",
                                yscale="linear",
                                subplots=(nrows, ncols),
                                linewidth=1,
                                box_linewidth=1,
                                markersize=8,
                                y_axis_size=10,
                                x_axis_size=10,
                                shareRowLabel=True)
    
    for rowi, row in enumerate(plotList):
        for fi, figure in enumerate(row):
            for trace in figure["traceList"]:
                pp_main.add_yxList(y_vals=trace["Y"], 
                                   x_vals=trace["X"], 
                                   label=trace["legend"],
                                   converged=trace["converged"])
            pp_main.plot(ylabel=figure["ylabel"], 
                         xlabel=figure["xlabel"],
                         yscale=figure["yscale"])
                         
    # SAVE THE WHOLE PLOT
    if outdir is not None:
        pp_main.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pp_main.fig.suptitle("")

        fig_name = os.path.join(outdir, "%s.pdf" % (exp_name))
        dirname = os.path.dirname(fig_name)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)
        pp_main.fig.savefig(fig_name, dpi = 600)
    
    return pp_main.fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)

    args = parser.parse_args()

    # Plot experiments
    # ===================
    exp_list = []
    for exp_group_name in args.exp_group_list:
        exp_list = exp_configs.EXP_GROUPS[exp_group_name]
        fig = plot_exp_list(exp_list, args.savedir_base, outdir='docs',
                            exp_name=exp_group_name, 
                            plot_names=exp_configs.PLOT_NAMES[exp_group_name])
        plt.close()
        print(exp_group_name, 'saved.')

    



