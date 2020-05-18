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




