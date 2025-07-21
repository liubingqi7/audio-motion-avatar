#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# def ssim(img1, img2, window_size=11, size_average=True):
#     channel = img1.size(-3)
#     window = create_window(window_size, channel)

#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)

#     return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    '''
    img1: [B, T, H, W, C]
    img2: [B, T, H, W, C]
    '''
    B, T = img1.shape[:2]
    img1 = img1.reshape(-1, *img1.shape[2:]).permute(0, 3, 1, 2)  # [B*T, C, H, W]
    img2 = img2.reshape(-1, *img2.shape[2:]).permute(0, 3, 1, 2)  # [B*T, C, H, W]
    
    channel = img1.size(1)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class LPIPS(torch.nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.lpips = lpips.LPIPS(net='vgg').cuda()

    def forward(self, img_out, img_target, bbox=None):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]
        img_out = img_out * 2 - 1 # [0,1] -> [-1,1]
        img_target = img_target * 2 - 1 # [0,1] -> [-1,1]
        loss = self.lpips(img_out, img_target)
        return loss
    
from smplx.lbs import batch_rodrigues  # 或者 kornia 的 angle_axis_to_rotation_matrix

def rotation_geodesic_loss(rot_vec_pred, rot_vec_gt):
    """
    计算两个旋转序列之间的测地线距离损失
    - rot_vec_*: tensor of shape [B, ..., 3]，轴角表示
    自动把最后一维的所有 3D 向量展平成 N×3，再计算 geodesic loss。
    """
    assert rot_vec_pred.shape == rot_vec_gt.shape, \
        f"Shape mismatch: {rot_vec_pred.shape} vs {rot_vec_gt.shape}"
    assert rot_vec_pred.shape[-1] == 3, "最后一维必须为 3"

    # 展平所有 batch+关节 维度到一个 N
    pred_flat = rot_vec_pred.reshape(-1, 3)  # [N,3]
    gt_flat   = rot_vec_gt.reshape(-1, 3)    # [N,3]

    # 转旋转矩阵
    R_pred = batch_rodrigues(pred_flat)      # [N,3,3]
    R_gt   = batch_rodrigues(gt_flat)        # [N,3,3]

    # geodesic: θ = arccos((trace(R_pred^T R_gt) - 1) / 2)
    RT = torch.matmul(R_pred.transpose(1, 2), R_gt)  # [N,3,3]
    cos = (torch.diagonal(RT, dim1=1, dim2=2).sum(-1) - 1) / 2
    cos = torch.clamp(cos, -0.999, 0.999)
    angle = torch.acos(cos)  # [N]

    return angle.mean()      # 标量损失

def smplx_param_loss(pred_params, gt_params, weights=None):
    if weights is None:
        weights = {
            'betas': 1.0,
            'global_orient': 1.0,
            'body_pose': 1.0,
            'left_hand_pose': 1.0,
            'right_hand_pose': 1.0,
            'jaw_pose': 1.0,
            'leye_pose': 1.0,
            'reye_pose': 1.0,
            'expression': 1.0,
            'transl': 1.0
        }
    
    losses = {}
    total_loss = 0.0

    if 'betas' in pred_params and 'betas' in gt_params:
        l_shape = F.mse_loss(pred_params['betas'], gt_params['betas'])
        prior_betas = torch.mean(pred_params['betas']**2)
        losses['betas_mse'] = l_shape
        losses['betas_prior'] = prior_betas
        total_loss += weights['betas'] * l_shape
        total_loss += 0.01 * prior_betas

    for key in ['global_orient', 'body_pose',
                'left_hand_pose', 'right_hand_pose',
                'jaw_pose', 'leye_pose', 'reye_pose']:
        if key in pred_params and key in gt_params:
            l_geo = rotation_geodesic_loss(pred_params[key], gt_params[key])
            losses[f'{key}_geo'] = l_geo
            total_loss += weights.get(key, 1.0) * l_geo

    if 'expression' in pred_params and 'expression' in gt_params:
        l_expr = F.l1_loss(pred_params['expression'], gt_params['expression'])
        prior_expr = torch.mean(pred_params['expression']**2)
        losses['expression_l1'] = l_expr
        losses['expression_prior'] = prior_expr
        total_loss += weights['expression'] * l_expr
        total_loss += 0.01 * prior_expr

    if 'transl' in pred_params and 'transl' in gt_params:
        l_transl = F.smooth_l1_loss(pred_params['transl'], gt_params['transl'])
        losses['transl_smoothl1'] = l_transl
        total_loss += weights['transl'] * l_transl

    # print(f"losses: {losses}")

    return total_loss, losses
