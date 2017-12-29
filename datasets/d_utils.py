import numpy as np
from scipy.io import savemat, loadmat
from base import utils as ut 


def generate_dataset(root, name):
    if name == "isingSmall":
        nrows = 10 
        ncols = 10
        W, y = isingSmall(nrows, ncols)

    if name == "ising1":
        nrows = 50 
        ncols = 50
        W, y = ising1(nrows, ncols)

    if name == "ising2":
        nrows = 50 
        ncols = 50
        W, y = ising2(nrows, ncols)

    if name == "nearest1":
        W, y = nearest1(root)

    assert np.diag(W).sum() == 0

    # Divide the data to unlabeled and labeled indices
    unlabeled_indices = (y == 0).ravel()
    labeled_indices = (y != 0).ravel()

    n = y.size

    # Initialize A and b and fill their values based on the 
    # formulation 0.5 * \sum_{ij} Wij (xi-xj)**2
    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(n):
        W_mi = W[labeled_indices, i]
        b[i] = np.sum(W_mi * y[labeled_indices])

        for j in range(n):
            if i == j:
                A[i, j] = np.sum(W[:, i])
            else:
                A[i, j] = - W[i, j]

    # Grab only the unlabeled indices as the dataset
    A = A[unlabeled_indices][:, unlabeled_indices]
    b = b[unlabeled_indices]
    
    if name in ["nearest1"]:
        return A, b, {}
    return A, b, {"data_nrows":nrows, "data_ncols":ncols, "data_y":y}


def nearest1(root):
    np.random.seed(1)
    W_y = loadmat('%s/W_y.mat' % (root))
    W = W_y["W"]

    y = W_y["y"]
    y = y.ravel()

    assert np.array_equal(np.unique(W), [0,1])
    return W, y

def ising2(nrows=50, ncols=50):
    np.random.seed(1)
    # CREATE "W"
    n = ncols * nrows
    W = np.zeros((n, n))

    for i in range(n):
        r = int(i % nrows)
        c = int(i / nrows)

        Wval = 1e5
        # Case 0: Bottom-right node
        if c == (ncols - 1) and r == (nrows - 1):
            pass

        # Case 1: Node in Right most edge
        elif c == (ncols - 1) and r != (nrows - 1):
            W[i, i + 1] = Wval
 
        # Case 1: Node in Bottom edge
        elif r == (nrows - 1):
            W[i, i + nrows] = Wval

        # Case 3: Left-Middle Edge node
        else: 
            W[i, i + 1] = Wval
            W[i, i + nrows] = Wval
            
    
    # Make it undirected
    W = W + np.tril(W.T, -1)

    # Labels
    ind = np.random.choice(n, 100, replace=False)
    y = np.zeros(n)

    y[ind] = np.random.randn(100) * 10.

    return W, y