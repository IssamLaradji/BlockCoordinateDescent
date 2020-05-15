from haven import haven_utils as hu 


A = {'name':'A', 'loss':'ls'}
A_l1 = {'name':'A', 'loss':'lsl1nn'}
B = {'name':'B', 'loss':'lg'}
C = {'name':'C', 'loss':'sf'}
D = {'name':'D', 'loss':'bp'}
E = {'name':'E', 'loss':'bp'}

EXP_GROUPS = {}

EXP_GROUPS['fig4'] = {'dataset':[A, C, E],
                      'partition':["VB", "Sort"],
                      'selection':['Random', 'Cyclic', 'Lipschitz', 'GS', 'GSDHb', 'GSL'],
                      'update':['Lb'],
                      'loss':['ls', 'sf', 'bp'],
                      'block_size':5, 
                      'max_iters':500}

EXP_GROUPS['fig5'] = {'dataset':[A, C, E],
                      'partition':["VB", "Sort"],
                      'selection':['GSQ', 'GS', 'GSL', 'GSD', 'IHT', 'GSDHb'],
                      'update':['Hb'],
                      'block_size':5, 
                      'max_iters':500}

EXP_GROUPS['fig6a'] = {'dataset':D,
                      'partition':["VB"],
                      'selection':['RTree', 'GSTree', 'GSExactTree', 'RedBlackTree', 'TreePartitions'],
                      'update':['Hb'],
                      'block_size':-1, 
                      'max_iters':500}

EXP_GROUPS['fig6b'] = {'dataset': E,
      'partition':"VB",
      'selection':['RTree', 'GSTree', 'GSExactTree', 'RedBlackTree',
                    'TreePartitions', 'TreePartitionsRandom',
                    'RedBlackTreeRandom'],
     'update':'bpExact',
      'block_size':-1,
      'max_iter':500}

EXP_GROUPS['fig7'] = {'dataset': A_l1,
     'partition':["VB", "Sort"],
      'selection':"gsq-nn",
     'update':["qp-nn", "TMP-NN", "Lb-NN"],
     "l1": 50000, 
       'block_size':[5, 50, 100],
        'max_iter':500
        }

EXP_GROUPS['fig8'] = {'dataset': [A, B, C, D, E],
     'partition':["Sort", "VB"],
     'selection':["Cyclic", "Lipschitz", "Perm", "Random", "GS", "GSL", "GSDHb"],
     'update':"Lb",
     'block_size':[5, 50, 100],
     'max_iter':500}

EXP_GROUPS['fig9'] = {'dataset': [A, B, C, D, E],
        'max_iter':500, 
     'partition':"Order",
     'selection':["GS", 'GSL', 'Lipschitz', 'Random', 'Cyclic'],
     'update':["LA", "Lb"],
     'block_size':[5, 20, 50]}

EXP_GROUPS['fig10'] = {'dataset': ['A','B', 'C', 'D', 'E'],
        'loss':['ls', 'lg', 'sf', 'bp', 'bp'], 
        'max_iter':500, 
     'partition':['Sort', 'Order', 'Avg'],
     'selection':['GS', 'GSD', 'GSL'],
     'update':"Lb",
     'block_size':[5, 20, 50]}

EXP_GROUPS['fig11'] = {'dataset': [A, B, C, D, E],
     'partition':["Sort", "VB"],
     'selection':["GSQ", "GS", "GSL", "GSD", "IHT"],
     'update':["Hb"],
     'block_size':[5, 50, 100],
      'max_iter':500}

EXP_GROUPS['fig12'] = {'dataset': [B, C],
     'partition':["Sort", "VB"],
     'selection':["GSQ", 'GS', 'GSL', 'GSD', "IHT"],
     'update':"LS",
     'block_size':[5, 50, 100],   
     'max_iter':500}

EXP_GROUPS['fig13'] = {'dataset': A_l1,
     'partition':["VB", "Sort"],
     'selection':"Random",
     'update':["qp-nn", "TMP-NN", "Lb-NN"],
     'l1': 50000,
      'block_size':[5, 50, 100], 
     'max_iter':500}

   
EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}