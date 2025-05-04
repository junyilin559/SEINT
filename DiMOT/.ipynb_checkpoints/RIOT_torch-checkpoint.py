import torch
from .torch_utils import *

def RIOT_torch(
        X,
        Y,
        lp = 2,
        rep = 100,
        determin = False,
        rd_rad = None,
        softmax = False,
        maxed = False,
        sliced = True,
        reg = 0.01,\
        device = 'cuda'
):
    """
    :param X: 初始点云 n*p
    :param Y: 初始点云 m*q
    :param lp: distance 范数
    :param rep: reference distribution 数量
    :param determin: 是否选用确定的 reference distance
    :param rd_rad: reference distribution的范围
    :param softmax: 是否进行softmax
    :param maxed: 是否选择取max
    :param reg:(如果为sinkhorn)对应的reg设定
    :return: RIOT的loss以及对应coupling
    """

    """
    sliced type
    """
    X = X.to(device)
    Y = Y.to(device)
    n= X.shape[0]
    m= Y.shape[0]
    Xn = X
    Yn = Y
    # Xn = (X - X.mean(dim=0)) / ((X - X.mean(dim=0)).std())
    # Yn = (Y - Y.mean(dim=0)) / ((Y - Y.mean(dim=0)).std())

    vectors1 = []
    vectors2 = []

    plengthX = torch.norm(Xn, 2, dim=1)
    plengthY = torch.norm(Yn, 2, dim=1)

    if rd_rad == None:
        rd_rad = (plengthX.mean() + plengthY.mean() + plengthX.std() + plengthY.std())/2

    for k in range(rep):
        if (determin):
            torch.manual_seed(k)
        npoints = rd_rad * generate_points(n, device = device)
        length1 = get_length5(plengthX, npoints)
        length2 = get_length5(plengthY, npoints)
        vectors1.append(length1)
        vectors2.append(length2)
    X1 = torch.stack(vectors1, dim=1)
    Y1 = torch.stack(vectors2, dim=1)

    X_dist = torch.cdist(Xn, Xn, p=lp)
    Y_dist = torch.cdist(Yn, Yn, p=lp)
    X_dist = X_dist / X_dist.std()
    Y_dist = Y_dist / Y_dist.std()

    # ttfeature
    if(softmax):
        if(sliced):
            X2 = row_softmax(X_dist) @ X1
            Y2 = row_softmax(Y_dist) @ Y1
        else:
            X2 = row_softmax(X_dist) @ X1
            Y2 = row_softmax(Y_dist) @ Y1   
    else:
        if(sliced):
            X2 = X_dist @ X1
            Y2 = Y_dist @ Y1
        else:
            X2 = X_dist @ X1
            Y2 = Y_dist @ Y1
    #OT
    if(sliced):   
        X2 = torch.sort(X2, dim=0)[0]/X1.mean(axis=0)
        Y2 = torch.sort(Y2, dim=0)[0]/Y1.mean(axis=0)
        #loss
        if(maxed):
            loss = torch.max(torch.sum((X2 - Y2) ** 2,dim=0))
        else:
            loss = torch.sum((X2 - Y2) ** 2)
        P = 'Sliced version without coupling'
        #print(P)
    else:
        a = torch.ones(n, device=device) / n
        b = torch.ones(m, device=device) / m
        M_torch = torch.cdist(X2, Y2, p=2).to(device)
        M_torch_mean = M_torch.mean()
        gamma_torch, log_data = sinkhorn(a, b, M_torch/ M_torch_mean, reg = reg, method="sinkhorn", verbose=False, log=True)
        #loss
        loss = (gamma_torch * M_torch).sum()
        
    return P, loss

def RIOT_batch(X, Y, RD, Softmax = False, Maxed = False, rd_rad = None):
    """
    batch type
    """
    std0 = (X.std()+Y.std())/2
    Xn = X /std0.detach()#/ X.std()
    Yn = Y /std0.detach()#/ Y.std()

    plengthX = torch.norm(Xn, 2, dim=1)
    plengthY = torch.norm(Yn, 2, dim=1)
    if rd_rad == None:
        rd_rad = (plengthX.mean() + plengthY.mean() + plengthX.std() + plengthY.std())/2
    X1 = get_length_batch(plengthX, rd_rad*RD).T
    Y1 = get_length_batch(plengthY, rd_rad*RD).T


    if(Softmax):
        X_dist = torch.cdist(Xn, Xn, p=2)
        Y_dist = torch.cdist(Yn, Yn, p=2)
        X2 = row_softmax(X_dist) @ X1
        Y2 = row_softmax(Y_dist) @ Y1
    else:
        X_dist = torch.cdist(Xn, Xn, p=2)**2
        Y_dist = torch.cdist(Yn, Yn, p=2)**2
        X2 = X_dist @ X1
        Y2 = Y_dist @ Y1

    X2 = torch.sort(X2, dim=0)[0]/X1.mean(axis=0)
    Y2 = torch.sort(Y2, dim=0)[0]/Y1.mean(axis=0)
    #loss
    if(Maxed):
        loss = torch.max(torch.sum((X2 - Y2) ** 2,dim=0))
    else:
        loss = torch.sum((X2 - Y2) ** 2)
    return loss

def RIOT_sliced_batch(X, Y, rep = 30, rd_rad = 3, maxed = True):
    _, n, _ = X.shape
    device = X.device
    Rrference_Dist = generate_matrix(n, rep, device)
    batched_lfOT = vmap(RIOT_batch, in_dims=(0, 0, None))
    output = batched_lfOT(X, Y, Rrference_Dist, rd_rad = rd_rad, Maxed = maxed)/rep
    return output
    

