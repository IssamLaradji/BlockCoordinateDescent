import numpy as np
from scipy.io import savemat, loadmat
from base import utils as ut 
from . import d_utils as du

def load(name, path=""):
    np.random.seed(1)
    
    if name == "A":

        # Least squares dataset
        data = loadmat('%s/exp4.mat' % path)
        A, b = data['X'], data['y']
        b = b.ravel()

        A = A.astype(float)
        b = b.astype(float)

        return {"A":A, "b":b, "args":{}}  

    elif name == "B":
        # Logistic dataset
        bias = 1; scaling = 10; sparsity = 10; solutionSparsity = 0.1;
        n = 1000;
        p = 10000;
        X = np.random.randn(n,p)+bias;
        X = X.dot(np.diag(scaling* np.random.randn(p)))
        X = X * (np.random.rand(n,p) < (sparsity*np.log(n)/n));
        w = np.random.randn(p) * (np.random.rand(p) < solutionSparsity);

        y = np.sign(X.dot(w));
        y = y * np.sign(np.random.rand(n)-.1);


        return {"A": X, "b": y, "args":{}}


    elif name == "C":
        # Softmax dataset
        bias = 1 
        scaling = 10
        sparsity = 10
        n_classes = 50

        n = 1000; p = 1000;
        A = np.random.randn(n, p) + bias;

        A = np.dot(A, np.diag(scaling*np.random.randn(p)))
        A = A * (np.random.rand(n,p) < sparsity*np.log(n)/n);
        w = np.random.randn(p, n_classes);

        b = np.dot(A,w) 
        b += np.random.randn(*b.shape)
        b = np.argmax(b, 1)
                
        b = ut.to_categorical(b, n_classes)

        return {"A":A, "b":b, "args":{}}  

    elif name == "D":
        # Ising model
        A, b, dargs = du.generate_dataset(path, "ising2")
        assert np.linalg.eigh(A)[0].min() > 0

        
        return {"A":A,"b": b, "args":dargs}  

    elif name == "E":
        # Label Prop
        A, b, dargs = du.generate_dataset(path,"nearest1")
        assert np.linalg.eigh(A)[0].min() > 0

        
        return {"A":A,"b": b, "args":dargs} 
