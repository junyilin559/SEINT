import numpy as np
import scipy as sp

def SEINT(
        X,
        Y,
        rd = None,
        initial = True,
        lp = 2,
        rep = 200,
        maxed = False,
        determin = False,
        set_seed = None,
        rd_rad = None,
        acc = False
):
    """
    Implement SEINT/ISEINT distance for equal sample size and equal weights

    Parameters
    ----------
        X: Initial point cloud (n x p)
        Y: Initial point cloud (n x q)
        rd: Reference matrix (n x rep) (not None if 'determin' is True).
        initial: Whether to initialize the input data
        lp: Lp norm for distance calculation.
        rep: Number of reference distributions.
        maxed: Whether to take the maximum value for the loss.
        determin: Whether to use a deterministic reference distance.
        rd_rad: Maximum value for the randomly generated reference distances/lengths.
        acc: Whether to use acceleration techniques.

    Returns
    ----------
        The SEINT/ISEINT loss.
    """

    # Initialize
    n, p = np.shape(X)
    m, q = np.shape(Y)
    if(initial):
        Xn = (X - X.mean(axis=0)) / (X - X.mean(axis=0)).std()
        Yn = (Y - Y.mean(axis=0)) / (Y - Y.mean(axis=0)).std()
    else:
        Xn = X
        Yn = Y

    X_plength = np.linalg.norm(Xn, axis = 1, ord=lp)
    Y_plength = np.linalg.norm(Yn, axis = 1, ord=lp)

    if rd_rad == None:
        rd_rad = (X_plength.mean() + Y_plength.mean() + X_plength.std() + X_plength.std())/2

    if(determin):
        # step = int(2*n / rep)
        L = np.tril(np.ones((n, n), dtype=int))
        rd = np.hstack([L + np.sort(X_plength)[:, None], L + np.sort(Y_plength)[:, None]])#[:, ::step]
        
        rep = rd.shape[1]
    else:
        if(set_seed != None):
            rng = np.random.default_rng(set_seed)
            rd = rng.uniform(0, rd_rad, size=(n, rep))
        else:
            rd = rd_rad * np.random.rand(n, rep)

    # t1 = time.time()
    rd = np.sort(rd, axis = 0)
    X1 = np.abs(rd - np.sort(X_plength)[:, None])[np.argsort(np.argsort(X_plength))]
    Y1 = np.abs(rd - np.sort(Y_plength)[:, None])[np.argsort(np.argsort(Y_plength))]
    # t2 = time.time()
    # print(t2 - t1)
    

    if(acc):
        X_plength2 = X_plength**2
        Y_plength2 = Y_plength**2
        X_dist_scale = (-2 * (Xn.sum(axis = 0)**2).sum() + 2*n*X_plength2.sum())/n/n
        Y_dist_scale = (-2 * (Yn.sum(axis = 0)**2).sum() + 2*m*Y_plength2.sum())/m/m
        X2 = -2*Xn @ (Xn.T @ X1) + np.outer(X_plength2 , np.ones(n) @ X1) + np.outer(np.ones(n), X_plength2 @ X1)
        Y2 = -2*Yn @ (Yn.T @ Y1) + np.outer(Y_plength2 , np.ones(m) @ Y1) + np.outer(np.ones(m), Y_plength2 @ Y1)
        X2 = X2 / X_dist_scale
        Y2 = Y2 / Y_dist_scale
    else:
        X_dist = sp.spatial.distance.cdist(Xn, Xn, metric='minkowski', p = lp)**lp
        Y_dist = sp.spatial.distance.cdist(Yn, Yn, metric='minkowski', p = lp)**lp
        X_dist = X_dist / X_dist.mean()
        Y_dist = Y_dist / Y_dist.mean()
        # ttfeature
        X2 = X_dist @ X1
        Y2 = Y_dist @ Y1

    # print(X2.shape)
    # Loss calculation
    X2 = np.sort(X2, axis=0)/X1.sum(axis=0)
    Y2 = np.sort(Y2, axis=0)/Y1.sum(axis=0)
    if(maxed):
        loss_list = np.mean((X2 - Y2) ** 2,axis=0)
        loss = np.max(loss_list)
    else:
        loss = np.mean(np.mean((X2 - Y2) ** 2,axis=0))
    
    return loss
