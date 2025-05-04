import numpy as np
import ot


def generate_RD(num_points, num_clusters):
    cluster_centers = np.random.rand(num_clusters)
    std_dev = np.random.uniform(0, 0.05, num_clusters)
    cluster_sizes = np.random.multinomial(num_points, np.ones(num_clusters) / num_clusters)

    points = []
    for i in range(num_clusters):
        cluster_points = cluster_centers[i] + np.random.randn(cluster_sizes[i]) * std_dev[i]
        points.append(cluster_points)
    return np.concatenate(points)

def sort_and_match_single_index(vec1, vec2):
    # 检查长度是否相等
    if len(vec1) != len(vec2):
        raise ValueError("两个向量的长度必须相等")
    
    # 对 vec1 排序，获取排序索引
    sorted_indices_vec1 = np.argsort(vec1)
    sorted_indices_vec1 = np.argsort(sorted_indices_vec1)
    sorted_indices_vec2 = np.argsort(vec2)
    
    return sorted_indices_vec1, sorted_indices_vec2

def get_length5(plength, npoints):
    indices1, indices2 = sort_and_match_single_index(plength, npoints)
    length = np.abs(plength - npoints[indices2][indices1])
    return length

def row_softmax(matrix):
    row_max = np.max(matrix, axis=1, keepdims=True)
    exp_matrix = np.exp(matrix - row_max)
    #exp_matrix = matrix
    softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
    return softmax_matrix


def RIOT(
        X,
        Y,
        lp = 2,
        rep = 200,
        method = 'emd',
        determin = True,
        rd_rad = None,
        rd_peak = 3,
        softmax = False,
        maxed = False,
        sliced = False,
        reg = 0.01,
        threepeaks = False
):
    """
    :param X: 初始点云 n*p
    :param Y: 初始点云 m*q
    :param lp: distance 范数
    :param rep: reference distribution 数量
    :param method: 最后coupling的计算方法
    :param determin: 是否选用确定的 reference distance
    :param rd_rad: reference distribution的范围
    :param rd_peak: reference distribution的峰数
    :param softmax: 是否进行softmax
    :param maxed: 是否选择取max
    :param reg:(如果为sinkhorn)对应的reg设定
    :return: RIOT的loss以及对应coupling
    """

    # Initialize
    n, p = np.shape(X)
    m, q = np.shape(Y)
    Xn = (X - X.mean(axis=0)) / (X - X.mean(axis=0)).std()
    Yn = (Y - Y.mean(axis=0)) / (Y - Y.mean(axis=0)).std()

    # OT length
    X_plength = np.linalg.norm(Xn, axis = 1)
    Y_plength = np.linalg.norm(Yn, axis = 1)
    vectors1 = []
    vectors2 = []
    if rd_rad == None:
        rd_rad = (X_plength.mean() + Y_plength.mean() + 3*X_plength.std() + 3*X_plength.std())/2
        print(rd_rad)
    for k in range(rep):
        if (determin):
            np.random.seed(k)
        if (threepeaks):
            npoints = rd_rad * generate_RD(n, rd_peak)
        else:
            npoints = rd_rad * np.random.rand(n)
        length1 = get_length5(X_plength, npoints)
        length2 = get_length5(Y_plength, npoints)
        vectors1.append(length1)
        vectors2.append(length2)
    X1 = np.vstack(vectors1).T
    Y1 = np.vstack(vectors2).T

    # Distance
    X_dist = ot.dist(Xn, Xn, 'euclidean')**lp
    Y_dist = ot.dist(Yn, Yn, 'euclidean')**lp
    X_dist = X_dist / X_dist.std()
    Y_dist = Y_dist / Y_dist.std()

    # ttfeature
    if(softmax):
        X2 = row_softmax(X_dist) @ X1
        Y2 = row_softmax(Y_dist) @ Y1

    else:
        X2 = X_dist @ X1
        Y2 = Y_dist @ Y1


    # OT
    if(sliced):
        X2_indices = np.argsort(X2, axis=0)
        Y2_indices = np.argsort(Y2, axis=0)   
        X2 = np.sort(X2, axis=0)/X1.mean(axis=0)
        Y2 = np.sort(Y2, axis=0)/Y1.mean(axis=0)
        col_losses = np.sum((X2 - Y2) ** 2, axis=0)
        #loss
        if(maxed):
            max_col_idx = np.argmax(col_losses)
            loss = np.sqrt(col_losses[max_col_idx]/((n+m)/2)**2)
            X2_orig_indices = X2_indices[:, max_col_idx]
            Y2_orig_indices = Y2_indices[:, max_col_idx]
            P = np.zeros((n,m), dtype=float)
            P[X2_orig_indices, Y2_orig_indices] = 1/n
        else:
            loss = np.sqrt(np.mean(col_losses)/((n+m)/2)**2)
            P = 'Sliced version without coupling'
    else:
        if((method == 'sinkhorn') | (method == 'emd')):
            a = np.ones(n) / n
            b = np.ones(m) / m
            C = ot.dist(X2, Y2, 'euclidean')
            C = C / C.mean()
        if(method == 'sinkhorn'):
            P = ot.sinkhorn(a, b, C, reg=reg, stopThr=1e-9)
        elif(method == 'emd'):
            P = ot.emd(a, b, C)
        # loss
        loss = (C*P).sum()
    
    return P, loss