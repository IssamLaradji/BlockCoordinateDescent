import numpy as np

def get_tree_slow(sorted_indices, adj):
    block = []       
    treeNumber = {}
    nTrees = 0
    for coordinate in sorted_indices:
      neighbors = np.nonzero(adj[coordinate])[0]
      nn = np.intersect1d(neighbors, block)
      neighTrees = set()

      continueFlag = False

      for neigh in nn:
        if treeNumber[neigh] not in neighTrees:
          neighTrees.add(treeNumber[neigh])
        else:
          continueFlag = True              
          break

        
      if continueFlag:              
          continue
      
      # if nn.size > 2:


      #   continue 

      if nn.size == 0:
         treeNumber[coordinate] = nTrees
         nTrees += 1
      else:

        treeNumber[coordinate] = nTrees

        for neigh in nn:
          treeID = treeNumber[neigh]
          for tree_ in treeNumber:
            if treeNumber[tree_] == treeID:
              treeNumber[tree_] = nTrees

            treeNumber[neigh] = nTrees
        nTrees += 1

      block += [coordinate]


    #block = np.array(block)[:block_size**3] 
    block = np.array(block)
    block.sort()  

    return block


def get_tp_indices(nrows, ncols, y):
    white = [np.arange(0, ncols*nrows, nrows)]
    black = [np.arange(nrows-1, ncols*nrows, nrows)]

    for c in range(ncols):
      if c % 2 == 0:
        white += [np.arange(c*nrows, (c+1)*nrows-1)]
      else:
        black += [np.arange(c*nrows+1, (c+1)*nrows)]

    white =  np.unique(np.hstack(white))
    black =  np.unique(np.hstack(black))
    
    assert black.size + white.size == (nrows * ncols)
    assert np.array_equal(np.unique(np.hstack([black, white])), np.arange(nrows*ncols))
    
    labeled = np.where(y != 0)[0]

    black = np.setdiff1d(black, labeled)
    white = np.setdiff1d(white, labeled)

    assert white.size + black.size == (y == 0).sum()

    unlabeled = np.where(y == 0)[0]

    gl2loc = {}
    for gl, loc in zip(unlabeled, np.arange(unlabeled.shape[0])):
      gl2loc[gl] = loc

    black = np.array([gl2loc[b] for b in black])
    white = np.array([gl2loc[w] for w in white])

    return {"black":black, "white":white}

def get_rb_indices(nrows, ncols, y):
    red = []
    black = []

    for c in range(ncols):
      odds = np.arange(1+c*nrows, (c+1)*nrows, 2)
      evens = np.arange(0+c*nrows, (c+1)*nrows, 2)

      if c % 2 == 0:
        red += [evens] 
        black += [odds]

      else:
        red += [odds] 
        black += [evens]


    red =  np.hstack(red)
    black =  np.hstack(black)


    assert black.size + red.size == (nrows * ncols)
    assert np.array_equal(np.unique(np.hstack([black, red])), np.arange(nrows*ncols))
    
    labeled = np.where(y != 0)

    black = np.setdiff1d(black, labeled)
    red = np.setdiff1d(red, labeled)

    assert red.size + black.size == (y == 0).sum()

    unlabeled = np.where(y == 0)[0]

    gl2loc = {}
    for gl, loc in zip(unlabeled, np.arange(unlabeled.shape[0])):
      gl2loc[gl] = loc

    black = np.array([gl2loc[b] for b in black])
    red = np.array([gl2loc[r] for r in red])


    return {"red":red, "black":black}