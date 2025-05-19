import numpy as np
import time
import matplotlib.pyplot as plt
import ot 
import scipy as sp
import sys
sys.path.append("C:/Users/xuedu/Desktop/DiMOT/SEINT-main/SEINT-main/SE(p) invariance/RISGW")### change your path
from risgw import risgw_gpu
import torch
sys.path.append("C:/Users/xuedu/Desktop/DiMOT/SEINT-main/SEINT-main")### change your path
from SEINT.SEINT_numpy import SEINT
from matplotlib.lines import Line2D
# ---------- methods ----------

def EGW(X, Y): 
    D11 = sp.spatial.distance.cdist(X, X)
    D12 = sp.spatial.distance.cdist(Y, Y)
    D11 /= D11.max()
    D12 /= D12.max()
    p = ot.unif(len(X))
    q = ot.unif(len(Y))
    dist = ot.gromov.entropic_gromov_wasserstein(
        D11, D12, p, q, loss_fun="square_loss", epsilon=0.01, log=True, verbose=False
    )[1]["gw_dist"]       
    return dist

def RISGW(X, Y):
    n, p = np.shape(X)
    nproj1 =  round(np.log10(n)*10)
    xi = torch.from_numpy(X).to(torch.float32)
    xj = torch.from_numpy(Y).to(torch.float32)
    dist = risgw_gpu(xi, xj, device='cpu', nproj=nproj1)
    return dist

def GW(X, Y):
    C3 = sp.spatial.distance.cdist(X, X)
    C4 = sp.spatial.distance.cdist(Y, Y)
    C3 /= C3.max()
    C4 /= C4.max()
    p = ot.unif(len(X))
    q = ot.unif(len(Y))
    dist = ot.gromov.gromov_wasserstein(
        C3, C4, p, q, "square_loss", verbose=False, log=True
    )[1]["gw_dist"]
    return dist

def SEINT_Func(X, Y):
    dist = SEINT(X,Y,  maxed=True,set_seed = 42, acc = True)
    return dist

def ISEINT_Func(X, Y):
    dist = SEINT(X,Y, maxed=False,set_seed = 42, acc = True)
    return dist
def sinkhorn(X, Y, reg=1e-5, p=2):
    a = ot.unif(len(X))         
    b = ot.unif(len(Y))
    M = ot.dist(X, Y, metric='euclidean') ** p  
    return ot.sinkhorn2(a, b, M, reg) ** (1.0 / p)

def emd(X, Y, p=2):
    a = ot.unif(len(X))
    b = ot.unif(len(Y))
    M = ot.dist(X, Y, metric='euclidean') ** p
    return ot.emd2(a, b, M) ** (1.0 / p)


METHODS = {
    "GW":     GW,
    "RISGW":      RISGW,
    "SINKHORN":        sinkhorn,
    "EMD": emd,
    "SEINT":         SEINT_Func,
    "ISEINT":  ISEINT_Func,
}

def make_cov_matrices(theta: float, d: int):
    """
    Σ_X  = diag(3 I₂ , I_{d-2})
    Σ_Y  = diag(3 I₂ + 3 θ B₂ , I_{d-2}),  B₂ = [[0,1],[1,0]]
    """
    # Σ_X
    Sigma_X = np.eye(d)
    Sigma_X[:2, :2] = 3 * np.eye(2)

    # Σ_Y
    B2 = np.array([[0., 1.],
                   [1., 0.]])
    Sigma_block = 3 * np.eye(2) + 3 * theta * B2
    Sigma_Y = np.eye(d)
    Sigma_Y[:2, :2] = Sigma_block
    return Sigma_X, Sigma_Y

def sample_gaussian(cov: np.ndarray, n: int,seed):
    np.random.seed(seed)
    return np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=n)

# ---------- EXP1 ----------
def experiment_theta_loss(theta_list, n=200, d=50, n_trials=10):
    """
    return:
        losses[method]  shape=(len(theta_list),)
    """
    losses_mean = {m: [] for m in METHODS}

    for theta in theta_list:
        trial_losses = {m: [] for m in METHODS}

        for _ in range(n_trials):
            Σ_X, Σ_Y = make_cov_matrices(theta, d)
            X = sample_gaussian(Σ_X, n,_*10)
            Y = sample_gaussian(Σ_Y, n,_*10)

            for name, func in METHODS.items():
                trial_losses[name].append(func(X, Y))

        for name in METHODS:
            losses_mean[name].append(np.mean(trial_losses[name]))

    return losses_mean  # dict → list

# ---------- EXP2 ----------
def experiment_n_time(n_list, theta=0.1, d=4, n_trials=10):
    """
    return:
        times_mean[method], times_std[method]  shape=(len(n_list),)
    """
    times_mean = {m: [] for m in METHODS}
    times_std  = {m: [] for m in METHODS}

    Σ_X_full, Σ_Y_full = make_cov_matrices(theta, d)  # θ 固定

    for n in n_list:
        trial_times = {m: [] for m in METHODS}

        for _ in range(n_trials):
            X = sample_gaussian(Σ_X_full, n,_*10)
            Y = sample_gaussian(Σ_Y_full, n,_*10)
            for name, func in METHODS.items():
                t0 = time.perf_counter()
                _  = func(X, Y)
                t1 = time.perf_counter()
                trial_times[name].append(t1 - t0)

        for name in METHODS:
            trial_times_arr = np.array(trial_times[name])
            times_mean[name].append(trial_times_arr.mean())
            times_std[name].append(trial_times_arr.std())

    return times_mean, times_std

####main
# ===== exp 1 =====
theta_vals = np.linspace(-1.0, 1.0, 9)           
loss_results = experiment_theta_loss(theta_vals)
# ===== exp 2 =====
n_vals = (10 ** np.arange(1.0, 4.1, 0.5)).astype(int)  
time_mean, time_std = experiment_n_time(n_vals, theta=0.1)


size = 17
# -------------------------------------------------PLOT
color_cycle = [
    "black", "dimgray",          # GW, RISGW
    "tab:blue", "deepskyblue",   # SINKHORN, EMD
    "tab:red", "indianred"       # SEINT, ISEINT
]
marker_cycle    = ["o", "s", "D", "^", "v", "*"]
linestyle_cycle = ["-", "--", "-.", ":", "-", "--"]

# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
fig.subplots_adjust(left=0.07, bottom=0.08, right=0.78, top=0.95)

# ========= (a) Distances vs theta =========
ax_a = axes[0]
for idx, (name, vals) in enumerate(loss_results.items()):
    y = vals if name == "SINKHORN" else vals / max(vals)
    ax_a.plot(theta_vals, y,
              color=color_cycle[idx],
              marker=marker_cycle[idx],
              linestyle=linestyle_cycle[idx],
              linewidth=1.8,
              markersize=5)
ax_a.tick_params(axis='both', labelsize=size)
ax_a.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax_a.set_xlabel(r'$\theta$',fontsize=size)
ax_a.set_ylabel('Distance',fontsize=size)
ax_a.set_title('(a)',fontsize=size)
ax_a.grid(True, linestyle=':', linewidth=0.6)

# ========= (b) Time vs n =========
ax_b = axes[1]
for idx, name in enumerate(METHODS):
    m  = time_mean
    sd = time_std
    ax_b.plot(n_vals, np.log10(m),
              color=color_cycle[idx],
              marker=marker_cycle[idx],
              linestyle=linestyle_cycle[idx],
              linewidth=1.8,
              markersize=5)
    ax_b.fill_between(n_vals,
                      np.log10(m - sd),
                      np.log10(m + sd),
                      color=color_cycle[idx],
                      alpha=0.18)
ax_b.tick_params(axis='both', labelsize=size)
ax_b.set_xlabel(r'$\log_{10}(n)$',fontsize=size)
ax_b.set_ylabel(r'$\log_{10}(Time)$',fontsize=size)
ax_b.set_title('(b)',fontsize=size)
ax_b.grid(True, linestyle=':', linewidth=0.6)

# -------------------------------------------------
METHODS = {
    "GW":     GW,
    "RISGW":      RISGW,
    "SINKHORN":        sinkhorn,
    "EMD": emd,
    "SEINT":         SEINT_Func,
    "ISEINT":  ISEINT_Func,
}
method_order = ["GW", "RISGW", "Sinkhorn", "EMD", "SEINT", "ISEINT"]
idx_map = {m: i for i, m in enumerate(method_order)}
proposed = ["SEINT", "ISEINT"]
other    = ["GW", "RISGW", "Sinkhorn", "EMD"]

handles_prop = [Line2D([], [], 
                       color=color_cycle[idx_map[m]],
                       marker=marker_cycle[idx_map[m]],
                       linestyle=linestyle_cycle[idx_map[m]],
                       linewidth=1.8, markersize=5, label=m)
                for m in proposed]

handles_other = [Line2D([], [], 
                        color=color_cycle[idx_map[m]],
                        marker=marker_cycle[idx_map[m]],
                        linestyle=linestyle_cycle[idx_map[m]],
                        linewidth=1.8, markersize=5, label=m)
                 for m in other]
legend1 = ax_b.legend(handles=handles_prop, title="Our Methods",
                      loc='upper left', bbox_to_anchor=(1.02, 1.02),
                      frameon=True,fontsize=15)
legend2 = ax_b.legend(handles=handles_other, title="Other Methods",
                      loc='upper left', bbox_to_anchor=(1.02, 0.62),
                      frameon=True,fontsize=15)
ax_b.add_artist(legend1) 
legend1.get_title().set_fontsize(15)
legend2.get_title().set_fontsize(15)


plt.tight_layout()
plt.savefig("figure.png", dpi=300)
plt.show()