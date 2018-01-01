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


class ForestBuilder:
  def __init__(self, ind=None, A=None):
    self.block = []       
    self.treeNumber = {}
    self.nTrees = 0
    if ind is not None:
      self.add2forest(ind, A)


  def checkForestProperty(self, ind, A):
    neighbors = np.nonzero(A[ind])[0]
    blockNeighs = np.intersect1d(neighbors, self.block)
    neighTrees = set()

    addNode = True

    for neigh in blockNeighs:
      if self.treeNumber[neigh] not in neighTrees:
        neighTrees.add(self.treeNumber[neigh])
      else:
        addNode = False              
        break

    return {"addNode":addNode, "blockNeighs":blockNeighs}

  def add2forest(self, ind, A):
    meta = self.checkForestProperty(ind, A)

    if meta["addNode"] is False:
      return False

    blockNeighs = meta["blockNeighs"]

    if blockNeighs.size == 0:
       self.treeNumber[ind] = self.nTrees
       self.nTrees += 1
    else:
      self.treeNumber[ind] = self.nTrees

      for neigh in blockNeighs:
        treeID = self.treeNumber[neigh]

        for tree_ in self.treeNumber:
          if self.treeNumber[tree_] == treeID:
            self.treeNumber[tree_] = self.nTrees

          self.treeNumber[neigh] = self.nTrees

      self.nTrees += 1

    self.block += [ind]

    return True

#### GENERAL GRAPH
def get_tp_general_graph(W, L=None):
  forestDict = {}

  n_nodes = W.shape[0]
  nodeIndices = np.arange(n_nodes)
  if L is not None:
    nodeIndices = np.argsort(L)[::-1]
  for i in nodeIndices:
    availForests = {}

    for f in forestDict:
      forest =  forestDict[f]
      if forest.checkForestProperty(i, W)["addNode"]:
        availForests[f] = len(forest.block)

    if len(availForests) == 0:
      # Create new color
      f = len(forestDict)
      forestDict[f] = ForestBuilder(ind=i,A=W)


    else:
      # Use existing color
      #f = min(availForests, key=availForests.get)
      f = min(availForests.keys())
      forestDict[f].add2forest(i, W)

  # SANITY CHECK
  alls = []
  blocks = []
  for f in forestDict:
    blocks += [forestDict[f].block]
    alls += forestDict[f].block

  alls = np.array(alls)

  assert np.unique(alls).size == alls.size
  assert alls.size == n_nodes
  #
  return tuple([np.array(b) for b in blocks])


def isForest(Wb):
  n = Wb.shape[0]

  Laplacian = Wb.copy()
  Laplacian[np.diag_indices(n)] = Wb.sum(1)
  
  LHS = 0.5*np.trace(Laplacian)
  RHS = np.linalg.matrix_rank(Laplacian)

  return LHS == RHS

def get_rb_general_graph(W, L=None):
  colorDict = {}
  blockDict = {}

  n_nodes = W.shape[0]
  node2color = np.ones(n_nodes) * -1
  nodeIndices = np.arange(n_nodes)
  if L is not None:
    nodeIndices = np.argsort(L)[::-1]
  for i in nodeIndices:
    neigh = np.where(W[i] != 0)[0]

    # Get neighbor colors that are not -1
    neighColors = np.unique(node2color[neigh])

    palette = {c:colorDict[c] for c in colorDict if c not in neighColors}

    if len(palette) == 0:
      # Create new color
      c = len(colorDict)
      
      colorDict[c] = 1
      blockDict[c] = [i]

      node2color[i] = c

    else:
      # Use existing color
      c = min(palette, key=palette.get)

      colorDict[c] += 1
      blockDict[c] += [i]

      node2color[i] = c
  
  # SANITY CHECK
  alls = []
  
  for v in blockDict.values():
    alls += v

  alls = np.array(alls)

  assert np.unique(alls).size == alls.size
  assert alls.size == n_nodes
  #

  return tuple([np.array(b) for b in blockDict.values()])


#### LATTICE GRAPH
def get_tp_indices(nrows, ncols):

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
    
    #return {"black":black, "white":white}
    return (black, white)



def remove_labeled_nodes(graphBlocks, y):
  labeled = np.where(y != 0)[0]
  unlabeled = np.where(y == 0)[0]

  gl2loc = {}
  for gl, loc in zip(unlabeled, np.arange(unlabeled.shape[0])):
    gl2loc[gl] = loc

  new_graphBlocks = []

  for block in graphBlocks:
    new_block = np.setdiff1d(block, labeled)
    new_block = np.array([gl2loc[b] for b in new_block])

    #return {"black":black, "white":white}
    new_graphBlocks += [new_block]

  return tuple(new_graphBlocks)



def get_rb_indices(nrows, ncols):
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
    
    #return {"red":red, "black":black}
    return (red, black)