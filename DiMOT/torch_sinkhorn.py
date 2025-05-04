import torch
import warnings
import math

def sinkhorn(a, b, M, reg, method="sinkhorn", numItermax=1000, stopThr=1e-9,
             verbose=False, log=False, warn=True, warmstart=None, **kwargs):
    """
    求解带熵正则化的最优运输问题，返回运输矩阵或损失（视具体函数而定）。

    Parameters
    ----------
    a : torch.Tensor, shape (dim_a,)
        源分布（要求非负并归一化）
    b : torch.Tensor, shape (dim_b,)
        目标分布（要求非负并归一化）
    M : torch.Tensor, shape (dim_a, dim_b)
        代价矩阵
    reg : float
        正则化系数 > 0
    method : str, optional
        使用的求解方法，可选：'sinkhorn', 'sinkhorn_log',
        'greenkhorn', 'sinkhorn_stabilized', 'sinkhorn_epsilon_scaling'
    numItermax : int, optional
        最大迭代次数
    stopThr : float, optional
        收敛阈值
    verbose : bool, optional
        是否打印迭代信息
    log : bool, optional
        是否记录日志（返回 log 字典）
    warn : bool, optional
        若不收敛是否发出警告
    warmstart : tuple, optional
        若提供则作为初始的对偶变量（或 u,v 初始化）
    **kwargs :
        其他参数传递给对应方法

    Returns
    -------
    根据具体方法返回：
      - 运输矩阵 gamma，或最优运输损失
      - 如果 log=True，则额外返回 log 字典
    """
    method = method.lower()
    if method == "sinkhorn":
        return sinkhorn_knopp(a, b, M, reg, numItermax, stopThr, verbose, log, warn, warmstart, **kwargs)
    # elif method == "sinkhorn_log":
    #     return sinkhorn_log(a, b, M, reg, numItermax, stopThr, verbose, log, warn, warmstart, **kwargs)
    # elif method == "greenkhorn":
    #     return greenkhorn(a, b, M, reg, numItermax, stopThr, verbose, log, warn, warmstart)
    # elif method == "sinkhorn_stabilized":
    #     return sinkhorn_stabilized(a, b, M, reg, numItermax, tau=1e3, stopThr=stopThr,
    #                                warmstart=warmstart, verbose=verbose, log=log, warn=warn, **kwargs)
    # elif method == "sinkhorn_epsilon_scaling":
    #     return sinkhorn_epsilon_scaling(a, b, M, reg, numItermax, stopThr=stopThr,
    #                                     warmstart=warmstart, verbose=verbose, log=log, warn=warn, **kwargs)
    else:
        raise ValueError("Unknown method '{}'.".format(method))


# def sinkhorn2(a, b, M, reg, method="sinkhorn", numItermax=1000, stopThr=1e-9,
#               verbose=False, log=False, warn=False, warmstart=None, **kwargs):
#     """
#     与 sinkhorn 类似，但返回最优运输损失（不含熵项），即 <gamma*, M>。

#     Parameters
#     ----------
#     a, b, M, reg, method, numItermax, stopThr, verbose, log, warn, warmstart
#         同 sinkhorn 函数

#     Returns
#     -------
#     loss : torch.Tensor 或 float
#         最优运输损失
#     若 log=True，则同时返回 log 字典
#     """
#     # 判断 b 是否多维（多目标情况）
#     if b.dim() < 2:
#         if method.lower() == "sinkhorn":
#             res = sinkhorn_knopp(a, b, M, reg, numItermax, stopThr, verbose, log, warn, warmstart, **kwargs)
#         elif method.lower() == "sinkhorn_log":
#             res = sinkhorn_log(a, b, M, reg, numItermax, stopThr, verbose, log, warn, warmstart, **kwargs)
#         elif method.lower() == "sinkhorn_stabilized":
#             res = sinkhorn_stabilized(a, b, M, reg, numItermax, stopThr, warmstart=warmstart,
#                                       verbose=verbose, log=log, warn=warn, **kwargs)
#         else:
#             raise ValueError("Unknown method '{}'.".format(method))
#         if log:
#             gamma, log_data = res
#             return torch.sum(gamma * M), log_data
#         else:
#             gamma = res
#             return torch.sum(gamma * M)
#     else:
#         # 多目标情况下直接调用对应方法
#         if method.lower() == "sinkhorn":
#             return sinkhorn_knopp(a, b, M, reg, numItermax, stopThr, verbose, log, warn, warmstart, **kwargs)
#         elif method.lower() == "sinkhorn_log":
#             return sinkhorn_log(a, b, M, reg, numItermax, stopThr, verbose, log, warn, warmstart, **kwargs)
#         elif method.lower() == "sinkhorn_stabilized":
#             return sinkhorn_stabilized(a, b, M, reg, numItermax, stopThr, warmstart=warmstart,
#                                        verbose=verbose, log=log, warn=warn, **kwargs)
#         else:
#             raise ValueError("Unknown method '{}'.".format(method))




def sinkhorn_knopp(
    a: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    reg: float,
    numItermax: int = 1000,
    stopThr: float = 1e-9,
    verbose: bool = False,
    log: bool = False,
    warn: bool = False,
    warmstart=None,
    **kwargs,
):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix (PyTorch version).

    This function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (dim_a, dim_b) cost matrix
    - :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})` is the entropic regularization term
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target histograms that both sum to 1

    The algorithm used for solving the problem is the classic Sinkhorn-Knopp
    matrix scaling algorithm as proposed in [2]_.

    Parameters
    ----------
    a : torch.Tensor, shape (dim_a,)
        Samples weights in the source domain (must be nonnegative, sum to 1)
    b : torch.Tensor, shape (dim_b,) or shape (dim_b, n_hists)
        Samples in the target domain. If `b` is a 2D tensor (dim_b, n_hists),
        then it solves multiple transport problems in parallel with the same cost M.
    M : torch.Tensor, shape (dim_a, dim_b)
        Loss (cost) matrix
    reg : float
        Regularization term > 0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True
    warn : bool, optional
        If True, raises a warning if the algorithm doesn't converge
    warmstart : tuple of torch.Tensor, optional
        Initialization of dual potentials. If provided, they are assumed to be log-scaled
        (u0, v0) = (log_u, log_v), so we do `u = exp(u0)`, `v = exp(v0)` as the initial guess.
        If None, we use uniform initialization.
    **kwargs :
        Absorbs any extra parameters (for compatibility)

    Returns
    -------
    gamma : torch.Tensor
        The optimal transportation matrix, shape (dim_a, dim_b), if b is 1D.
        If b is 2D, then by default this function returns only the losses (one per column of b).
    log_dict : dict (optional)
        Returned only if log == True, containing "err", "niter", "u", "v" etc.

    References
    ----------
    .. [2] M. Cuturi, "Sinkhorn Distances : Lightspeed Computation of Optimal Transport",
       Advances in Neural Information Processing Systems (NIPS) 26, 2013.
    """

    # 1) 处理 a, b 的空输入（与原版一致）
    if a.numel() == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0], dtype=M.dtype, device=M.device)
    if b.numel() == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1], dtype=M.dtype, device=M.device)

    # 2) 判断 b 是否多目标
    if b.dim() > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    dim_a = a.shape[0]
    dim_b = b.shape[0] if not n_hists else b.shape[0]

    # 3) 若需要记录日志则初始化日志字典
    if log:
        log_dict = {"err": []}
    else:
        log_dict = None

    # 4) 初始化 u, v
    #    如果 warmstart 不为空，则假设传入的是 (log_u, log_v)，需要先取 exp
    #    如果是多目标，u, v 也相应是 2D
    if warmstart is None:
        if n_hists > 0:
            u = torch.ones(dim_a, n_hists, dtype=M.dtype, device=M.device) / dim_a
            v = torch.ones(dim_b, n_hists, dtype=M.dtype, device=M.device) / dim_b
        else:
            u = torch.ones(dim_a, dtype=M.dtype, device=M.device) / dim_a
            v = torch.ones(dim_b, dtype=M.dtype, device=M.device) / dim_b
    else:
        # warmstart: (log_u, log_v)
        log_u, log_v = warmstart
        u = torch.exp(log_u)
        v = torch.exp(log_v)

    # 5) 计算核矩阵 K = exp(-M / reg)
    K = torch.exp(-M / reg)

    # 6) 计算 Kp = diag(1/a) * K，用于更新 u
    #   注意 a 的形状是 (dim_a,)，需要先 unsqueeze
    #   如果 b 是多目标，a 仍是一维，这里与原版逻辑一致
    Kp = (1.0 / a).unsqueeze(1) * K

    err = float('inf')
    # 7) 主循环
    for ii in range(numItermax):
        uprev = u.clone()
        vprev = v.clone()

        # 7.1) KtransposeU = K^T * u
        KtransposeU = torch.matmul(K.t(), u)
        # 7.2) v = b / KtransposeU
        v = b / (KtransposeU + 1e-16)
        # 7.3) u = 1 / (Kp * v)
        #      其中 Kp = diag(1/a) * K
        #      => Kp v = (1/a) * (K v)
        #      => u = 1 / (Kp v)
        # 如果是多目标，Kp.shape = (dim_a, dim_b), v.shape=(dim_b, n_hists)，结果 (dim_a, n_hists)
        # 如果是单目标，则 (dim_a,) 维度对应
        uv = torch.matmul(Kp, v)
        u = 1.0 / (uv + 1e-16)

        # 7.4) 检查数值是否出现 0 或 inf 或 nan
        #      如果出现则回退到上一步
        if (
            torch.any(KtransposeU == 0)
            or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
            or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))
        ):
            warnings.warn(f"Warning: numerical errors at iteration {ii}")
            u = uprev
            v = vprev
            break

        # 7.5) 每 10 次迭代检查一次收敛
        if ii % 10 == 0:
            if n_hists > 0:
                # 多目标情况: tmp2 = einsum("ik,ij,jk->jk", u, K, v)
                # 这里手动写成 torch.einsum
                tmp2 = torch.einsum("ik,ij,jk->jk", u, K, v)
            else:
                # 单目标情况: tmp2 = einsum("i,ij,j->j", u, K, v)
                tmp2 = torch.einsum("i,ij,j->j", u, K, v)

            # err = ||tmp2 - b||  (1范数)
            # b 若是多目标 (dim_b, n_hists)，则 tmp2.shape=(dim_b, n_hists)
            # b 若是单目标 (dim_b,)
            err = torch.norm(tmp2 - b, p=1)

            if log_dict is not None:
                # 记录误差
                log_dict["err"].append(err.item())

            # 如果误差小于阈值则退出
            if err.item() < stopThr:
                break

            if verbose:
                if ii % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err"))
                    print("-" * 19)
                print("{:5d}|{:8e}|".format(ii, err.item()))

    else:
        # 如果 for-else 没有 break，则说明循环结束仍未收敛
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )

    # 8) 如果要 log，则存储最终迭代次数和 u,v
    if log_dict is not None:
        log_dict["niter"] = ii
        log_dict["u"] = u
        log_dict["v"] = v

    # 9) 根据是否多目标返回不同结果
    if n_hists > 0:
        # 多目标时返回的不是 OT 矩阵，而是多个 transport losses
        # 原版: res = nx.einsum("ik,ij,jk,ij->k", u, K, v, M)
        # => 这里对应 torch.einsum("ik,ij,jk,ij->k", u, K, v, M)
        #   其中 u.shape=(dim_a, n_hists), K.shape=(dim_a, dim_b),
        #   v.shape=(dim_b, n_hists), M.shape=(dim_a, dim_b)
        #   => result shape=(n_hists,)
        res = torch.einsum("ik,ij,jk,ij->k", u, K, v, M)
        if log_dict is not None:
            return res, log_dict
        else:
            return res
    else:
        # 单目标时返回 OT 矩阵
        gamma = u.unsqueeze(1) * K * v.unsqueeze(0)
        if log_dict is not None:
            return gamma, log_dict
        else:
            return gamma
