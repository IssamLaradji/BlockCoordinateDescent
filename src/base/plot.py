import numpy as np 
import pylab as plt
from haven import haven_utils as hu
from skimage.io import imread 
from . import pretty_plot

name_list = ['a','b','c', 'd', 'e','f', 'g','h', 'i','j','k','l','m','n','o','p']
def plot(plotList, expName, savedir_base, ratio=1, 
        axFontSize=14,
        axTickSize=14, legend_size=10, 
        linewidth=1, markersize=10,
        box_linewidth=1,fname=None):

    # CREATE PLOT OBJECT
    nrows = len(plotList)
    ncols = len(plotList[0])

    # Main plot
    pp_main = pretty_plot.PrettyPlot(title=expName, 
    axFontSize=axFontSize,
                                axTickSize=axTickSize,
                                legend_size=legend_size,
                                #ylabel="Loss", 
                                #xlabel="Iterations with %d-sized blocks" % bs,
                                figsize=(5*ncols,4*nrows),
                                legend_type="line",
                                yscale="linear",
                                subplots=(nrows, ncols),
                                linewidth=linewidth,
                                box_linewidth=box_linewidth,
                                markersize=markersize,
                                shareRowLabel=True)
    # n_plot_id = 0
    # for rowi, row in enumerate(plotList):
    #     for fi, figure in enumerate(row):
    #         pp_sub = pretty_plot.PrettyPlot(
    #             axFontSize=axFontSize,
    #                                 axTickSize=axTickSize,
    #                                     figsize=(5,4),
    #                                     legend_size=legend_size,
    #                                     legend_type="line",
    #                                     yscale="linear",
    #                                     subplots=(1, 1),
    #                                     linewidth=linewidth,
    #                             box_linewidth=box_linewidth,
    #                             markersize=markersize,
    #                                     shareRowLabel=True)

            
               
    #         for trace in figure["traceList"]:
                
    #             pp_main.add_yxList(y_vals=trace["Y"], 
    #                                x_vals=trace["X"], 
    #                                label=trace["legend"],
    #                                converged=trace["converged"])

    #             pp_sub.add_yxList(y_vals=trace["Y"], 
    #                                x_vals=trace["X"], 
    #                                label=trace["legend"],
    #                                converged=trace["converged"])

    #         pp_main.plot(ylabel=figure["ylabel"], 
    #                      xlabel=figure["xlabel"],
    #                      yscale=figure["yscale"])



    #         pp_sub.plot(ylabel=figure["ylabel"], 
    #                     xlabel=figure["xlabel"],
    #                     yscale=figure["yscale"])

    #         if fname is not None:
    #             pp_sub.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #             # axName = "%s/pdf_subplots/%s/%d_%d.pdf" % (savedir_base, expName, rowi, fi)
    #             axName = "%s/pdf_subplots/%s/%s.pdf" % (savedir_base, expName, name_list[n_plot_id])
    #             ut.create_dirs(axName)
    #             pp_sub.fig.savefig(axName, dpi = 600)

    #             # pp_sub.fig.suptitle(expName)

    #             # axName = "%s/png_subplots/%s/%d_%d.png" % (savedir_base, expName, rowi, fi)
    #             # ut.create_dirs(axName)
    #             # pp_sub.fig.savefig(axName)
    #             fi += 1
    #             n_plot_id += 1
    #         #pp_sub.fig.close()


    # SAVE THE WHOLE PLOT
    if fname is not None:
        pp_main.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # figName = "%s/png_plots/%s.png" % (savedir_base, expName)
        # ut.create_dirs(figName)
        # pp_main.fig.savefig(figName)

        pp_main.fig.tight_layout()
        pp_main.fig.suptitle("")

        figName = "%s/pdf_plots/%s.pdf" % (savedir_base, expName)
        hu.create_dirs(figName)
        pp_main.fig.savefig(figName, dpi = 600)
    
    return pp_main.fig


        
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
