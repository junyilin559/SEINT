import numpy as np
import scipy as sp
from .numpy_utils import *

def RIOT(
        X,
        Y,
        lp = 2,
        rep = 200,
        method = 'maxed',
        return_coupling = False,
        determin = True,
        rd_rad = None,
        maxed = False,
):
    """
    :param X: 初始点云 n*p
    :param Y: 初始点云 n*q
    :param lp: distance 范数
    :param rep: reference distribution 数量
    :param method: 最后coupling的计算方法
    :param determin: 是否选用确定的 reference distance
    :param maxed: 是否选择取max
    :return_coupling: 是否返回coupling
    :return: RIOT的loss以及对应coupling
    """

    # Initialize
    n, p = np.shape(X)
    m, q = np.shape(Y)
    Xn = (X - X.mean(axis=0)) / (X - X.mean(axis=0)).std()
    Yn = (Y - Y.mean(axis=0)) / (Y - Y.mean(axis=0)).std()

    # OT length
    X_plength = np.linalg.norm(Xn, axis = 1, ord=lp)
    Y_plength = np.linalg.norm(Yn, axis = 1, ord=lp)
    vectors1 = []
    vectors2 = []
    if rd_rad == None:
        rd_rad = (X_plength.mean() + Y_plength.mean() + X_plength.std() + X_plength.std())/2
        
    for k in range(rep):
        if (determin):
            np.random.seed(k)
        if (maxed):
            npoints = rd_rad * np.random.rand(n)
        else:
            npoints = rd_rad * generate_RD(n, 3)
        length1 = get_length5(X_plength, npoints)
        length2 = get_length5(Y_plength, npoints)
        vectors1.append(length1)
        vectors2.append(length2)
    X1 = np.vstack(vectors1).T
    Y1 = np.vstack(vectors2).T

    # Distance
    X_dist = sp.spatial.distance.cdist(Xn, Xn, metric='minkowski', p = lp)**lp
    Y_dist = sp.spatial.distance.cdist(Yn, Yn, metric='minkowski', p = lp)**lp
    X_dist = X_dist / X_dist.std()
    Y_dist = Y_dist / Y_dist.std()

    # t-tfeature
    X2 = X_dist @ X1
    Y2 = Y_dist @ Y1

    # OT
    X2 = np.sort(X2, axis=0)/X1.sum(axis=0)
    Y2 = np.sort(Y2, axis=0)/Y1.sum(axis=0)
    #loss
    if(maxed):
        loss_list = np.mean((X2 - Y2) ** 2,axis=0)
        loss = np.max(loss_list)
        index = np.argmax(loss_list)
        ot_index = sort_and_match_single_index(X2[:,index], Y2[:,index])
        return ot_index, loss
    else:
        loss = np.mean(np.mean((X2 - Y2) ** 2,axis=0))
        P = 'Expected version without coupling'
    
    return P, loss