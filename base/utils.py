import os
import numpy as np
import json
import shlex
import pandas as pd
import glob, logging, sys
from shutil import copyfile
from matplotlib.transforms import Bbox
import itertools
import time 
import pickle
import pylab as plt
from scipy.misc import imsave, imread

def visplot(fig, win="tmp"):
    import visdom
    fig.savefig("tmp.jpg")
    img = imread("tmp.jpg").transpose((2,0,1))
    print(img.shape)
    vis = visdom.Visdom(port=1111)

    options = dict(title=win)
    vis.images(img, win=win, env='main', opt=options) 
    
    plt.close()


def load_pkl(fname):
    with open(fname, "rb") as f:        
        return pickle.load(f)


def save_pkl(fname, dict):
    create_dirs(fname)
    with open(fname, "wb") as f: 
        pickle.dump(dict, f)

def fname2dict():
    pass 

def dict2fname():
    pass
    
def timeit(fun, *args):
    start = time.time()
    ret = fun(*args)
    end = time.time()

    print({"result": ret, "time": end-start})

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def descFunc(p_rules, s_rules, u_rules, plot_names=None):
    if plot_names is None:
        rule2name = {}
    else:
        rule2name = {pn.split(":")[0]:pn.split(":")[1] 
                     for pn in plot_names}

    def get_name(r):
        if r in rule2name:
            return rule2name[r]
        else:
            return r 

    desc = ""
    if len(u_rules) == 1:
        desc += "%s-" % get_name(u_rules[0])

    if len(s_rules) == 1:
        desc += "%s-" % get_name(s_rules[0])

    if len(p_rules) == 1:
        desc += "%s-" % get_name(p_rules[0])


    desc = desc[:-1] 
    desc += ("\nNames: %s" % str(plot_names))

    return desc

def legendFunc(p, s, u, p_rules, s_rules, u_rules, plot_names=None):
    if plot_names is None:
        rule2name = {}
    else:
        rule2name = {pn.split(":")[0]:pn.split(":")[1] 
                     for pn in plot_names}

    def get_name(r):
        if r in rule2name:
            return rule2name[r]
        else:
            return r 

    legend = ""
    if len(u_rules) > 1:
        legend += "%s-" % get_name(u)

    if len(s_rules) > 1:
        legend += "%s-" % get_name(s)

    if len(p_rules) > 1:
        legend += "%s-" % get_name(p)


    return legend[:-1]

def prune_rules():
    pass 

    
def print2file(statement, logname):
    print(statement)

def get_logger(logname, backup=False, append=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(1)

    formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # create a file handler
    if backup and os.path.exists(logname):
        # BACKUPS 
        backup = "logs/backups/"
        backup += logname.split("/")[-1].replace(".log", "")

        n = len(glob.glob(backup + "*"))
        copyfile(logname, 
                 "%s_%d.log" % (backup, n))
    if append:
        handler = logging.FileHandler(logname, mode='a')
    else:
        handler = logging.FileHandler(logname, mode='w')
    handler.setLevel(1)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create output handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(1)
    logger.addHandler(handler)

    return logger 

def remove_alias(aRules):
    rules = []
    for r in aRules:
        if "-" in r:
            rules += [r.strip().split("-")[0]]
        else:
            rules += [r]

    return rules

def get_name_dict(rules, name_dict):
    rules_new = []
    for r in rules:
        if "(" in r and ")" in r:
            rr = r.replace(" ","")
            rr = rr.replace(")","")
            rr = rr.replace("(","")
            rr = rr.split(",")
            
            name_dict[rr[0]] = rr[1]
            rules_new += [rr[0]]
        else:
            name_dict[r] = r
            rules_new += [r]

    return name_dict, rules_new

def dict2str(dict):
    string = ""

    for k in dict:
        string += "%s: %.3f" % (k, float(dict[k]))
        string += " - "

    return string[:-3]

def parseArg_json(name, parser, fname="exps.json"):
    # LOAD EXPERIMENTS
    with open(fname) as data_file:
        exp_dict = json.loads(data_file.read())
    
    argString = exp_dict[name]
    
    if isinstance(argString, list):
      argString = " ".join(argString)

    io_args = parser.parse_args(shlex.split(argString))

    return io_args

def load_results(fpath, info):

    reset = info["reset"]
    
    if os.path.exists(fpath + ".csv") and not reset:
        results = read_csv(fpath)       
        stored_info = read_json(fpath)
        info_equal = dict_equal(info, stored_info)

        if info_equal:
            return results

    return pd.DataFrame()
            
def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass  
def save_csv(path, csv_file):
    create_dirs(path)
    csv_file.to_csv(path + ".csv", index=False) 

    print(("csv file saved in %s" % (path)))

def save_json(path, dictionary):
    create_dirs(path)
    with open(path + ".json" , 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)
    print(("JSON saved in %s" % path))

              
def read_json(path):
    with open(path + '.json') as data_file:    
        dictionary = json.load(data_file)
    
    return dictionary

def read_csv(path):
    csv = pd.read_csv(path + ".csv")
    return csv

def dict_equal(d1, d2):
    for key in d1:

        if key in ["p", "s", "u", "selection_rules", "partition_rules", "update_rules", "ylimIgnore", "block_size", "yloss", "scale", 
        "test","minLoss"]:
            continue
        if key not in d2:
            return False
        v1 = d1[key] 
        v2 = d2[key]

        if v1 != v2:
            print(("Diff (%s): %s != %s" % (key, v1, v2)))
            import pdb; pdb.set_trace()  # breakpoint 556b58d2 //

            return False

    return True


### MISC ALGORITHMS
def gradient_approx(x, f_func, n_params=3, eps=1e-7):
  e = np.zeros(x.size)
  
  gA = np.zeros(n_params)
  for n in range(n_params):
    
    e[n] = 1.
    val = f_func(x + e * np.complex(0, eps))
    gA[n] = np.imag(val) / eps
    e[n] = 0

  return gA

def hessian_approx(x, g_func, n_params=3, eps=1e-6):
  hA = np.zeros((n_params, n_params))
  for j in range(n_params):
    f_func = lambda x: g_func(x)[j]
    hA[j] = gradient_approx(x, f_func, n_params, eps=1e-6)

  return hA




def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y