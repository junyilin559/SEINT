import sys
sys.path.append('..')
sys.path.append('./LinearGromov')
sys.path.append('./SGW/lib')
import FastGromovWass
from risgw import risgw_gpu
from SEINT.SEINT_numpy import *

import torch
import numpy as np
import scipy as sp
import ot
import trimesh

def rotation_matrix_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def rotation_matrix_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotation_matrix_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotate_vector(v, theta_x, theta_y, theta_z):
    Rx = rotation_matrix_x(theta_x)
    Ry = rotation_matrix_y(theta_y)
    Rz = rotation_matrix_z(theta_z)
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    return np.dot(v, R)

filename = "data/models/horse-gallop/horse-gallop-reference.obj"
reference = np.array(trimesh.load(filename, force='mesh').vertices)
reference = rotate_vector(reference, 45, 45, 135)

## SEINT/ISEINT
Loss_seint = np.zeros([4, 48])
Loss_iseint = np.zeros([4, 48])

for i in range(1, 49):
    filename = f"data/models/horse-gallop/horse-gallop-{i:02d}.obj"
    y_dist = np.array(trimesh.load(filename, force='mesh').vertices)
    max_loss0 = SEINT(reference, y_dist, rep = 20, acc = True)
    max_loss1 = SEINT(reference[:,[0,1]], y_dist, rep = 20, acc = True)
    max_loss2 = SEINT(reference[:,[0,2]], y_dist, rep = 20, acc = True)
    max_loss3 = SEINT(reference[:,[1,2]], y_dist, rep = 20, acc = True)
    mean_loss0 = SEINT(reference, y_dist, maxed = True, rep = 20, acc = True)
    mean_loss1 = SEINT(reference[:,[0,1]], y_dist, maxed = True, rep = 20, acc = True)
    mean_loss2 = SEINT(reference[:,[0,2]], y_dist, maxed = True, rep = 20, acc = True)
    mean_loss3 = SEINT(reference[:,[1,2]], y_dist, maxed = True, rep = 20, acc = True)
    
    Loss_seint[0, i-1] = max_loss0
    Loss_seint[1, i-1] = max_loss1
    Loss_seint[2, i-1] = max_loss2
    Loss_seint[3, i-1] = max_loss3

    Loss_iseint[0, i-1] = mean_loss0
    Loss_iseint[1, i-1] = mean_loss1
    Loss_iseint[2, i-1] = mean_loss2
    Loss_iseint[3, i-1] = mean_loss3

Loss_seint = Loss_seint / Loss_seint.max(axis = 1)[:, np.newaxis]
Loss_iseint = Loss_iseint / Loss_iseint.max(axis = 1)[:, np.newaxis]
np.save('Loss_seint.npy', Loss_seint)
np.save('Loss_iseint.npy', Loss_iseint)

## RISGW
Loss_risgw = np.zeros([4, 48])
nproj=20
device = 'cuda'
for i in range(1, 49):
    filename = f"data/models/horse-gallop/horse-gallop-{i:02d}.obj"
    y_dist = np.array(trimesh.load(filename, force='mesh').vertices)

    risgw_loss0 = risgw_gpu(torch.from_numpy(reference).to(torch.float32).to(device),
                                 torch.from_numpy(y_dist).to(torch.float32).to(device),
                                 device,nproj=nproj, max_iter=100, lr=0.01)
    risgw_loss1 = risgw_gpu(torch.from_numpy(reference[:,[0,1]]).to(torch.float32).to(device),
                                 torch.from_numpy(y_dist).to(torch.float32).to(device),
                                 device,nproj=nproj, max_iter=100, lr=0.01)
    risgw_loss2 = risgw_gpu(torch.from_numpy(reference[:,[0,2]]).to(torch.float32).to(device),
                                 torch.from_numpy(y_dist).to(torch.float32).to(device),
                                 device,nproj=nproj, max_iter=100, lr=0.01)
    risgw_loss3 = risgw_gpu(torch.from_numpy(reference[:,[1,2]]).to(torch.float32).to(device),
                                 torch.from_numpy(y_dist).to(torch.float32).to(device),
                                 device,nproj=nproj, max_iter=100, lr=0.01)
    
    Loss_risgw[0, i-1] = risgw_loss0
    Loss_risgw[1, i-1] = risgw_loss1
    Loss_risgw[2, i-1] = risgw_loss2
    Loss_risgw[3, i-1] = risgw_loss3

Loss_risgw = Loss_risgw / Loss_risgw.max(axis = 1)[:, np.newaxis]
np.save('Loss_risgw.npy', Loss_risgw)

## GW/EGW
C1 = sp.spatial.distance.cdist(reference, reference)
C2 = sp.spatial.distance.cdist(reference[:, [0,1]], reference[:, [0,1]])
C3 = sp.spatial.distance.cdist(reference[:, [0,2]], reference[:, [0,2]])
C4 = sp.spatial.distance.cdist(reference[:, [1,2]], reference[:, [1,2]])

C1 /= C1.max()
C2 /= C2.max()
C3 /= C3.max()
C4 /= C4.max()

p = ot.unif(len(reference))
q = ot.unif(len(reference))
reg = 5 * 1e-3


print(1)
Loss_gw = np.zeros([4, 48])
Loss_egw = np.zeros([4, 48])

for i in range(1, 49):
    filename = f"data/models/horse-gallop/horse-gallop-{i:02d}.obj"
    y_dist = np.array(trimesh.load(filename, force='mesh').vertices)
    C_y = sp.spatial.distance.cdist(y_dist, y_dist)
    C_y /= C_y.max()

    egw_loss0, _, _, _, _ = FastGromovWass.GW_entropic_distance(
        C1, C_y, reg, p, q,
        Init="lower_bound", seed_init=49, I=100, delta_sin=1e-3, num_iter_sin=10000, 
        lam_sin=0, LSE=False, time_out=5000,)
    egw_loss1, _, _, _, _ = FastGromovWass.GW_entropic_distance(
        C2, C_y, reg, p, q,
        Init="lower_bound", seed_init=49, I=100, delta_sin=1e-3, num_iter_sin=10000, 
        lam_sin=0, LSE=False, time_out=5000,)
    egw_loss2, _, _, _, _ = FastGromovWass.GW_entropic_distance(
        C3, C_y, reg, p, q,
        Init="lower_bound", seed_init=49, I=100, delta_sin=1e-3, num_iter_sin=10000, 
        lam_sin=0, LSE=False, time_out=5000,)
    egw_loss3, _, _, _, _ = FastGromovWass.GW_entropic_distance(
        C4, C_y, reg, p, q,
        Init="lower_bound", seed_init=49, I=100, delta_sin=1e-3, num_iter_sin=10000, 
        lam_sin=0, LSE=False, time_out=5000,)

    _, gw_loss0 = ot.gromov.gromov_wasserstein(
        C1, C_y, p, q, "square_loss", verbose=True, log=True
    )
    _, gw_loss1 = ot.gromov.gromov_wasserstein(
        C2, C_y, p, q, "square_loss", verbose=True, log=True
    )
    _, gw_loss2 = ot.gromov.gromov_wasserstein(
        C3, C_y, p, q, "square_loss", verbose=True, log=True
    )
    _, gw_loss3 = ot.gromov.gromov_wasserstein(
        C4, C_y, p, q, "square_loss", verbose=True, log=True
    )

    Loss_gw[0, i - 1] = gw_loss0['gw_dist']
    Loss_gw[1, i - 1] = gw_loss1['gw_dist']
    Loss_gw[2, i - 1] = gw_loss2['gw_dist']
    Loss_gw[3, i - 1] = gw_loss3['gw_dist']

    Loss_egw[0, i-1] = egw_loss0
    Loss_egw[1, i-1] = egw_loss1
    Loss_egw[2, i-1] = egw_loss2
    Loss_egw[3, i-1] = egw_loss3

Loss_gw = Loss_gw / Loss_gw.max(axis = 1)[:, np.newaxis]
Loss_egw = Loss_egw / Loss_egw.max(axis = 1)[:, np.newaxis]
np.save('Loss_egw.npy', Loss_egw)
np.save('Loss_gw.npy', Loss_gw)
