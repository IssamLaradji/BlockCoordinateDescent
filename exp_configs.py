from haven import haven_utils as hu 
import numpy as np

A = {'name':'A', 'loss':'ls'}
A_l1 = {'name':'A', 'loss':'lsl1nn'}
B = {'name':'B', 'loss':'lg'}
C = {'name':'C', 'loss':'sf'}
D = {'name':'D', 'loss':'bp'}
E = {'name':'E', 'loss':'bp'}

EXP_GROUPS = {}

EXP_GROUPS['fig4a'] = {'dataset':[A],
                      'partition':["VB", "Sort"],
                      'selection':['Random', 'Cyclic', 'Lipschitz', 'GS', 'GSDHb', 'GSL'],
                      'update':['Lb'],
                      'block_size':5, 
                      'max_iters':500}

EXP_GROUPS['fig4b'] = {'dataset':[C],
                      'partition':["VB", "Sort"],
                      'selection':['Random', 'Cyclic', 'Lipschitz', 'GS', 'GSDHb', 'GSL'],
                      'update':['Lb'],
                      'block_size':5, 
                      'max_iters':500}
                      
EXP_GROUPS['fig4c'] = {'dataset':[ E],
                      'partition':["VB", "Sort"],
                      'selection':['Random', 'Cyclic', 'Lipschitz', 'GS', 'GSDHb', 'GSL'],
                      'update':['Lb'],
                      'block_size':5, 
                      'max_iters':500}

EXP_GROUPS['fig5a'] = {'dataset':[A],
                      'partition':["VB", "Sort"],
                      'selection':['GSQ', 'GS', 'GSL', 'GSD', 'IHT', 'GSDHb'],
                      'update':['Hb'],
                      'block_size':5, 
                      'max_iters':500}
EXP_GROUPS['fig5b'] = {'dataset':[C],
                      'partition':["VB", "Sort"],
                      'selection':['GSQ', 'GS', 'GSL', 'GSD', 'IHT', 'GSDHb'],
                      'update':['Hb'],
                      'block_size':5, 
                      'max_iters':500}
EXP_GROUPS['fig5c'] = {'dataset':[E],
                      'partition':["VB", "Sort"],
                      'selection':['GSQ', 'GS', 'GSL', 'GSD', 'IHT', 'GSDHb'],
                      'update':['Hb'],
                      'block_size':5, 
                      'max_iters':500}
EXP_GROUPS['fig6a'] = {'dataset':D,
                      'partition':["VB"],
                      'selection':['RTree', 'GSTree', 'GSExactTree', 'RedBlackTree', 'TreePartitions'],
                      'update':['bpExact'],
                      'block_size':-1, 
                      'max_iters':500}

EXP_GROUPS['fig6b'] = {'dataset': E,
      'partition':"VB",
      'selection':['RTree', 'GSTree', 'GSExactTree', 'RedBlackTree',
                    'TreePartitions', 'TreePartitionsRandom',
                    'RedBlackTreeRandom'],
     'update':'bpExact',
      'block_size':-1,
      'max_iters':500}

EXP_GROUPS['fig7'] = {'dataset': A_l1,
     'partition':["VB", "Sort"],
      'selection':"gsq-nn",
     'update':["qp-nn", "TMP-NN", "Lb-NN"],
     "l1": 50000, 
       'block_size':[5, 50, 100],
        'max_iters':500
        }

EXP_GROUPS['fig8'] = {'dataset': [A, B, C, D, E],
     'partition':["Sort", "VB"],
     'selection':["Cyclic", "Lipschitz", "Perm", "Random", "GS", "GSL", "GSDHb"],
     'update':"Lb",
     'block_size':[5, 50, 100],
     'max_iters':500}

EXP_GROUPS['fig9'] = {'dataset': [A, B, C, D, E],
        'max_iters':500, 
     'partition':"Order",
     'selection':["GS", 'GSL', 'Lipschitz', 'Random', 'Cyclic'],
     'update':["LA", "Lb"],
     'block_size':[5, 20, 50]}

EXP_GROUPS['fig10'] = {'dataset': [A, B, C, D, E],
        'max_iters':500, 
     'partition':['Sort', 'Order', 'Avg'],
     'selection':['GS', 'GSD', 'GSL'],
     'update':"Lb",
     'block_size':[5, 20, 50]}

EXP_GROUPS['fig11'] = {'dataset': [A, B, C, D, E],
     'partition':["Sort", "VB"],
     'selection':["GSQ", "GS", "GSL", "GSD", "IHT"],
     'update':["Hb"],
     'block_size':[5, 50, 100],
      'max_iters':500}

EXP_GROUPS['fig12'] = {'dataset': [B, C],
     'partition':["Sort", "VB"],
     'selection':["GSQ", 'GS', 'GSL', 'GSD', "IHT"],
     'update':"LS",
     'block_size':[5, 50, 100],   
     'max_iters':500}

EXP_GROUPS['fig13'] = {'dataset': A_l1,
     'partition':["VB", "Sort"],
     'selection':"Random",
     'update':["qp-nn", "TMP-NN", "Lb-NN"],
     'l1': 50000,
      'block_size':[5, 50, 100], 
     'max_iters':500}

   
EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}

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


PLOT_NAMES = {}
PLOT_NAMES['fig4a'] = PLOT_NAMES['fig4b'] = PLOT_NAMES['fig4c'] = ['Sort:FB', 'GSDHb:GSL']
PLOT_NAMES['fig5a'] = PLOT_NAMES['fig5b'] =  PLOT_NAMES['fig5c'] =  ['Sort:FB', 'IHT:GSQ', 'GSDHb:GSL']
PLOT_NAMES['fig6a'] = ["GSExactTree:General" 
                              "TreePartitions:Tree Partitions",
                              "RedBlackTree:Red Black",
                              "GSTree:Greedy Tree",
                              "RTree:Random Tree"]
PLOT_NAMES['fig6b'] = ['GSExactTree:General',
     'TreePartitions:Tree Partitions Lipschitz',
          'RedBlackTree:Red Black Lipschitz',
               'GSTree:Greedy Tree',
          'RTree:Random Tree',
               'TreePartitionsRandom:Tree Partitions Order',
          'RedBlackTreeRandom:Red Black Order']
PLOT_NAMES['fig7'] = ['Sort:FB', 'qp-nn:PN', 'TMP-NN:TMP', 'Lb-NN:PG']
PLOT_NAMES['fig8'] = ['Sort:FB', 'Perm:Cyclic', 'GSDHb:GSL']
PLOT_NAMES['fig9'] = ['Sort:FB']
PLOT_NAMES['fig10'] =  None
PLOT_NAMES['fig11'] = ['Sort:FB', 'IHT:GSQ']
PLOT_NAMES['fig12'] = ['Sort:FB', 'IHT:GSQ']
PLOT_NAMES['fig13'] = ['Sort:FB', 'qp-nn:PN', 'TMP-NN:TMP', 'Lb-NN:PG']