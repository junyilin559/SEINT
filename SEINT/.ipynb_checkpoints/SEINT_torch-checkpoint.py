import torch
from .torch_utils import *

def SEINT_torch(
        X,
        Y,
        rd = None,
        initial = False,
        lp = 2,
        rep = 200,
        maxed = False,
        determin = False,
        set_seed = None,
        rd_rad = None,
        acc = False,
        device = 'cuda'
):
    """
    Torch implement SEINT/ISEINT distance for equal sample size and equal weights

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
        device: The desired device of calculation (e.g., 'cpu', 'cuda').

    Returns
    ----------
        The SEINT/ISEINT loss (a torch.Tensor).
    """

    # Initialize
    X = X.to(device)
    Y = Y.to(device)
    n= X.shape[0]
    m= Y.shape[0]
    if(initial):
        Xn = (X - X.mean(dim=0)) / ((X - X.mean(dim=0)).std())
        Yn = (Y - Y.mean(dim=0)) / ((Y - Y.mean(dim=0)).std())
    else:
        Xn = X
        Yn = Y
    
    vectors1 = []
    vectors2 = []
    plengthX = torch.norm(Xn, lp, dim=1)
    plengthY = torch.norm(Yn, lp, dim=1)
    if rd_rad == None:
        rd_rad = (plengthX.mean() + plengthY.mean() + plengthX.std() + plengthY.std())/2

    # PTD calculation
    if(determin):
        _, rep = rd.shape[1]
    else:
        if(set_seed != None):
            gen = torch.Generator(device=device)
            gen.manual_seed(set_seed)
            rd = rd_rad * torch.rand((n, rep), generator=gen, device=device)
        else:
            rd = rd_rad * torch.rand((n, rep), device=device)
    
    for k in range(rep):
        length1 = get_ptd(plengthX, rd[:, k])
        length2 = get_ptd(plengthY, rd[:, k])
        vectors1.append(length1)
        vectors2.append(length2)
    X1 = torch.stack(vectors1, dim=1)
    Y1 = torch.stack(vectors2, dim=1)

    if(acc):
        X_plength2 = plengthX**2
        Y_plength2 = plengthY**2
        X2 = -2 * torch.matmul(Xn, torch.matmul(Xn.T, X1)) + torch.outer(X_plength2, torch.matmul(torch.ones(n, device=device), X1)) + torch.outer(torch.ones(n, device=device), torch.matmul(X_plength2, X1))
        Y2 = -2 * torch.matmul(Yn, torch.matmul(Yn.T, Y1)) + torch.outer(Y_plength2, torch.matmul(torch.ones(m, device=device), Y1)) + torch.outer(torch.ones(m, device=device), torch.matmul(Y_plength2, Y1))
    else:
        X_dist = torch.cdist(Xn, Xn, p=lp)**lp
        Y_dist = torch.cdist(Yn, Yn, p=lp)**lp
        # X_dist = X_dist / X_dist.std() # (optional) additional normalization
        # Y_dist = Y_dist / Y_dist.std()
        X2 = X_dist @ X1
        Y2 = Y_dist @ Y1
            
    # Loss calculation
    X2 = torch.sort(X2, dim=0)[0]/X1.mean(axis=0)
    Y2 = torch.sort(Y2, dim=0)[0]/Y1.mean(axis=0)
    if(maxed):
        loss = torch.max(torch.sum((X2 - Y2) ** 2,dim=0))
    else:
        loss = torch.mean(torch.sum((X2 - Y2) ** 2,dim=0))
    return loss

def SEINT_batch(X, Y, RD, Maxed = False, rd_rad = None):
    """
    batch type
    """
    # Prevent numerical overflow
    std0 = (X.std()+Y.std())/2
    Xn = X /std0.detach()
    Yn = Y /std0.detach()

    plengthX = torch.norm(Xn, 2, dim=1)
    plengthY = torch.norm(Yn, 2, dim=1)
    if rd_rad == None:
        rd_rad = (plengthX.mean() + plengthY.mean() + plengthX.std() + plengthY.std())/2
    X1 = get_ptd_batch(plengthX, rd_rad*RD).T
    Y1 = get_ptd_batch(plengthY, rd_rad*RD).T

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
        loss = torch.mean(torch.sum((X2 - Y2) ** 2,dim=0))
    return loss

def SEINT_batch_vmap(X, Y, rep = 30, rd_rad = 3, maxed = True):
    _, n, _ = X.shape
    device = X.device
    Rrference_Dist = generate_matrix(n, rep, device)
    batched_SEINT = vmap(SEINT_batch, in_dims=(0, 0, None))
    output = batched_SEINT(X, Y, Rrference_Dist, rd_rad = rd_rad, Maxed = maxed)/rep
    return output
    

