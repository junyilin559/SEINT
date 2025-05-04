# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import torch
from utils.wasserstein import sliced_wasserstein_batch,sliced_gw_batch
import chamfer
from DiMOT.RIOT_numpy import *
from DiMOT.RIOT_torch import *
class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2))/2




class CD2_fused_RIOT(torch.nn.Module):
    f''' Chamfer Distance L2 + RIOT '''
    def __init__(self, ignore_zeros=False, use_riot= True, riot_rep=100, rd_rad = 3, maxed = True):#参数设置
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.use_riot = use_riot
        self.rd_rad = rd_rad
        self.maxed = maxed
        if use_riot:
            self.riot = RIOTDistance_sliced(rep=riot_rep, rd_rad = rd_rad, maxed = self.maxed)

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        #print(batch_size)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        chamfer_loss = torch.mean(dist1) + torch.mean(dist2)

        if self.use_riot:
            riot_loss = self.riot(xyz1, xyz2)/(35*2)      
            return 0.9* chamfer_loss +  0.1 * riot_loss
        else:
            return chamfer_loss


class RIOTDistance_sliced(torch.nn.Module):
    def __init__(self, rep=100,rd_rad = None,maxed = True):#参数设置
        super().__init__()
        self.rep = rep
        self.rd_rad = rd_rad
        self.maxed = maxed

    def forward(self, xyz1, xyz2):
        """
        xyz1: [B, N, 3]
        xyz2: [B, N, 3]
        return: scalar RIOT loss (float)
        """
        loss = torch.mean(RIOT_sliced_batch(xyz1,xyz2,rep = self.rep ,rd_rad = self.rd_rad, maxed = self.maxed))/7000
        return loss

class sliced_wasserstein(torch.nn.Module):
    def __init__(self, num_projections=100):  # 参数设置
        super().__init__()
        self.num_projections = num_projections

    def forward(self, xyz1, xyz2):
        """
        xyz1: [B, N, 3]
        xyz2: [B, N, 3]
        return: scalar RIOT loss (float)
        """
        loss = torch.mean(sliced_wasserstein_batch(xyz1, xyz2, num_projections=self.num_projections))
        return loss


class CD2_fused_SW(torch.nn.Module):
    f''' Chamfer Distance L2 + SW '''
    def __init__(self, ignore_zeros=False, num_projections=100):#参数设置
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.num_projections = num_projections
        self.sw = sliced_wasserstein(num_projections=self.num_projections)

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        #print(batch_size)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        chamfer_loss = torch.mean(dist1) + torch.mean(dist2)


        sw_loss = self.sw(xyz1, xyz2) * 3
        # print(sw_loss,'sw_loss')
        # print(chamfer_loss,'chamfer_loss')
        return 0.9* chamfer_loss +  0.1 * sw_loss

class sliced_gw(torch.nn.Module):
    def __init__(self, num_projections=100):  # 参数设置
        super().__init__()
        self.num_projections = num_projections

    def forward(self, xyz1, xyz2):
        """
        xyz1: [B, N, 3]
        xyz2: [B, N, 3]
        return: scalar RIOT loss (float)
        """
        loss = torch.mean(sliced_gw_batch(xyz1, xyz2, num_projections=self.num_projections))
        return loss

class CD2_fused_SGW(torch.nn.Module):
    f''' Chamfer Distance L2 + SW '''

    def __init__(self, ignore_zeros=False, num_projections=100):  # 参数设置
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.num_projections = num_projections
        self.sgw = sliced_gw(num_projections=self.num_projections)

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        # print(batch_size)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        chamfer_loss = torch.mean(dist1) + torch.mean(dist2)

        sgw_loss = self.sgw(xyz1, xyz2) /2000000
        # print(sgw_loss,'sw_loss')
        # print(chamfer_loss,'chamfer_loss')
        return 0.9 * chamfer_loss + 0.1 * sgw_loss
