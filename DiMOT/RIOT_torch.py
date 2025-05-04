import torch
from torch_sinkhorn import sinkhorn

def sort_and_match_single_index(vec1, vec2):
    # 检查长度是否相等
    if len(vec1) != len(vec2):
        raise ValueError("两个向量的长度必须相等")

    # 对 vec1 排序，获取排序索引
    sorted_indices_vec1 = torch.argsort(vec1)
    sorted_indices_vec1 = torch.argsort(sorted_indices_vec1)
    sorted_indices_vec2 = torch.argsort(vec2)

    return sorted_indices_vec1, sorted_indices_vec2


def get_length5(plength, npoints):
    indices1, indices2 = sort_and_match_single_index(plength, npoints)
    length = torch.abs(plength - npoints[indices2][indices1])
    return length


def row_softmax(matrix):
    row_max = torch.max(matrix, dim=1, keepdim=True)[0]
    exp_matrix = torch.exp(matrix - row_max)
    softmax_matrix = exp_matrix / torch.sum(exp_matrix, dim=1, keepdim=True)
    return softmax_matrix


def generate_points(n, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    positions = torch.rand(3, device=device)
    positions = torch.sort(positions).values
    random_indices = torch.randint(0, 3, (n,), device=device)  # 随机选择三个位置中的一个
    points = positions[random_indices] + torch.rand(n, device=device) * (
                positions[random_indices] - torch.roll(positions, shifts=1)[random_indices])
    return points


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
    

