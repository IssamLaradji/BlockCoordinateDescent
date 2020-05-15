import numpy as np
from scipy.io import savemat, loadmat
from src.base import utils as ut 
from . import tree_datasets

def load(name, path=""):
    np.random.seed(1)
    
    if name == "A":
        """
        
        * **Dataset A** consists of $A \in \mathbb{R}^{1000 \times 10000}$ and
        $b \in \mathbb{R}^{1000}$ and is used to evaluate optimizing the least
        square objective function:

                            0.5 * ||Ax - b||^2

        We construct $A$ as follows (matlab code included below): 


        1. initialize $A$ with values drawn independantly from the normal distribution
        $\mathcal{N}(0, 1)$ and add 1 to all its elements; 


            Matlab code:
            ------------

            n=1000; p=10000; bias=1
            X = randn(n, p) + bias

        2. multiply each column in $A$ by a value drawn from $\mathcal{N}(0,1)$ multiplied by 10;


            Matlab code:
            ------------

            scaling = 10;
            X = X*diag(scaling*randn(p,1));

        3. keep each element in $A$ as non-zero with probability $10 \cdot \log(n)/ n$
        where $n$ is the number of rows in $A$;

            Matlab code:
            ------------

            sparsity = 10;
            X = X .* (rand(n, p) < sparsity * log(n)/n)

        4. Let $w$ be a vector $\in \mathbb{R}^{10000}$ whose elements are drawn from the normal
        distribution $N(0, 1)$. Induce sparsity on $w$ by leaving each element as non-zero with 
        probability 0.1. 

            Matlab code:
            ------------
            
            solutionSparsity = 0.1;
            w = randn(p, 1) .* (rand(p,1) < solutionSparsity);

        5. Construct $b$ as:
            
            \begin{equation}
            b = Aw + \epsilon
            \end{equation}
            
            where $\epsilon$ is in $\mathbb{R}^{1000}$ where each entry is drawn
            from $N(0, 1)$.

            Matlab code:
            ------------
            y = X*w + randn(n,1)

        """

        data = loadmat('%s/exp4.mat' % path)
        A, b = data['X'], data['y']
        b = b.ravel()

        A = A.astype(float)
        b = b.astype(float)

        return {"A":A, "b":b, "args":{}}  

    elif name == "B":
        """

        **Dataset B** is used to evaluate the optimization of the logistic
        objective function where $A \in \mathbb{R}^{1000 \times 10000}$ and
        $b \in \{-1, 1\}^{1000}$. $A$ and $w$ are constructed the same way as
        for dataset A. $b$ is defined as:

        \begin{equation}
        b = sign(Aw)
        \end{equation}

        For each entry in $b$ we change the sign with probability $0.1$.

        """
        bias = 1; scaling = 10; 
        sparsity = 10; solutionSparsity = 0.1;
        n = 1000;
        p = 10000;
        A = np.random.randn(n,p)+bias;
        A = A.dot(np.diag(scaling* np.random.randn(p)))
        A = A * (np.random.rand(n,p) < (sparsity*np.log(n)/n));
        w = np.random.randn(p) * (np.random.rand(p) < solutionSparsity);

        b = np.sign(A.dot(w));
        b = b * np.sign(np.random.rand(n)-0.1);


        return {"A": A, "b": b, "args":{}}


    elif name == "C":
        """
        50-class Softmax dataset

        It is a 50 class dataset used to evaluate multi-class logistic
        function optimization. 

        1. Construct $A \in {1000, 1000}$  the same way as for dataset $A$ and $B$. 

        2. Construct a dense $w \in \mathbb{R}^{1000 \times 50}$ each element in 
           $w$ is drawn from N(0,1)
           
        3. Define $B = Aw$. 

        4. Add a value drawn from $N(0,1)$ to each element in $B$. 

        Note that $b \in \{0,1,2,...,49\}^{1000}$ 

        """
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
        """
        Lattice-Based Quadratic
        
        **Dataset D** is a dataset generated using a 50 by 50 lattice-structured
        dependency for evaluating message passing algorithms. We use the
        following quadratic labeling criterion.

        \begin{equation}
        \min_{y_i | i \notin S} \frac{1}{2} \sum_{i=1}^b \sum_{j=1}^n w_{ij} (y_i - y_j)^2
        \label{eq:quadratic}
        \end{equation}

        where y is our label vector (each $y_i$ is sampled from $N(0, 100)$), S is
        the set of labels that we do know and $w_{ij} \geq 0$ are the weights
        assigned to each $y_i$ describing how strongly we want the label $y_i$
        and $y_j$ to be similar.  We have $w_{ij} \in \{0, 10^5\}$. 

        We expressed this quadratic problem as a
        linear system $Ax=b$; where we labelled 100 points in our data. The
        resulting linear system has a matrix of size 2400 x 2400 while the
        number of neighbours of each node is at most 4.

        """
        A, b, dargs = tree_datasets.generate_datasets_D_or_E(path, "ising")
        
        # Check that it is positive definite
        assert np.linalg.eigh(A)[0].min() > 0

        
        return {"A":A,"b": b, "args":dargs}  

    elif name == "E":
        """
        Unstructured Quadratic

        **Dataset E** corresponds to a label propagation problem. For this
        dataset, we generate a 2000 node graph based on the two-moons dataset
        and randomly label 100 points in the data. We then connect each node
        to its 5 nearest neighbours. We use eq. \ref{eq:quadratic} as the criterion 
        for creating the dataset as a linear system $Ax=b$ where $A$ is of size 1900 x 1900.

        We set the non-zero values of W to 1.
        
        """
        A, b, dargs = tree_datasets.generate_datasets_D_or_E(path,"nearest")

        # Check that it is positive definite
        assert np.linalg.eigh(A)[0].min() > 0

        
        return {"A":A,"b": b, "args":dargs} 


