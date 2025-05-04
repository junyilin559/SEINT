import torch
from functorch import vmap

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
    positions = 0.9 * torch.rand(3, device=device) + 0.05
    positions = torch.sort(positions).values
    random_indices = torch.randint(0, 3, (n,), device=device)  # 随机选择三个位置中的一个
    points = positions[random_indices] + 0.05 * torch.rand(n, device=device)
    return points

def generate_matrix(n, k, device='cuda'):
    return torch.stack([generate_points(n, device) for _ in range(k)], dim=1)

def get_length_batch(plength, RD):
    device = plength.device
    batched_lfOT = vmap(get_length5, in_dims=(None, 0))
    output = batched_lfOT(plength, RD.T)
    return output

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
    output = batched_lfOT(X, Y, Rrference_Dist, rd_rad = rd_rad, maxed = maxed)/rep
    return output





