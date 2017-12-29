import numpy as np 
import pylab as plt

from base import utils as ut
from skimage.io import imread 
from base import pretty_plot



def plot(plotList, expName, path=""):

    # CREATE PLOT OBJECT
    nrows = len(plotList)
    ncols = len(plotList[0])

    # Main plot
    pp_main = pretty_plot.PrettyPlot(title=expName, 
                                ratio=0.5,
                                #ylabel="Loss", 
                                #xlabel="Iterations with %d-sized blocks" % bs,
                                figsize=(5*ncols,4*nrows),
                                legend_type="line",
                                yscale="linear",
                                subplots=(nrows, ncols),
                                shareRowLabel=True)
    
    for rowi, row in enumerate(plotList):
        for fi, figure in enumerate(row):
            pp_sub = pretty_plot.PrettyPlot(ratio=0.5,
                                        figsize=(5,4),
                                        legend_type="line",
                                        yscale="linear",
                                        subplots=(1, 1),
                                        shareRowLabel=True)

            
               
            for trace in figure["traceList"]:
                
                pp_main.add_yxList(y_vals=trace["Y"], 
                                   x_vals=trace["X"], 
                                   label=trace["legend"],
                                   converged=trace["converged"])

                pp_sub.add_yxList(y_vals=trace["Y"], 
                                   x_vals=trace["X"], 
                                   label=trace["legend"],
                                   converged=trace["converged"])

            pp_main.plot(ylabel=figure["ylabel"], 
                         xlabel=figure["xlabel"],
                         yscale=figure["yscale"])



            pp_sub.plot(ylabel=figure["ylabel"], 
                        xlabel=figure["xlabel"],
                        yscale=figure["yscale"])

            pp_sub.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            axName = "%s/pdf_subplots/%s/%d_%d.pdf" % (path, expName, rowi, fi)
            ut.create_dirs(axName)
            pp_sub.fig.savefig(axName, dpi = 600)

            pp_sub.fig.suptitle(expName)

            axName = "%s/png_subplots/%s/%d_%d.png" % (path, expName, rowi, fi)
            ut.create_dirs(axName)
            pp_sub.fig.savefig(axName)
            fi += 1
            #pp_sub.fig.close()


    # SAVE THE WHOLE PLOT
    pp_main.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figName = "%s/png_plots/%s.png" % (path, expName)
    ut.create_dirs(figName)
    pp_main.fig.savefig(figName)

    pp_main.fig.tight_layout()
    pp_main.fig.suptitle("")

    figName = "%s/pdf_plots/%s.pdf" % (path, expName)
    ut.create_dirs(figName)
    pp_main.fig.savefig(figName, dpi = 600)

    plt.close()


        
def vinit(imgList, wins=None):
    if not isinstance(imgList, list):
        imgList = [imgList]


    if wins is None:
        wins = np.arange(len(imgList))
        wins = list(map(str, list(wins)))

        wins = list(wins)

    if not isinstance(wins, list):
        wins = [wins]

    return imgList, wins
