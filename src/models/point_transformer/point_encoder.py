import torch
import torch.nn as nn
from .pointtransformer_v3 import PointTransformerV3
from omegaconf import DictConfig

class PTv3Encoder(nn.Module):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.point_transformer = PointTransformerV3(
            in_channels=cfg.input_dim, 
            stride=cfg.stride,
            enc_channels=cfg.enc_channels, 
            enc_depths=cfg.enc_depths, 
            dec_channels=cfg.dec_channels, 
            dec_depths=cfg.dec_depths, 
            enc_num_head=cfg.enc_num_head, 
            dec_num_head=cfg.dec_num_head, 
            enc_patch_size=cfg.enc_patch_size, 
            dec_patch_size=cfg.dec_patch_size, 
            enable_flash=cfg.enable_flash
        )

        self.grid_resolution = 100

    def forward(self, pts, feats):
        B, N, C = pts.shape
        feats = feats.reshape(B*N, -1)
        offset = torch.tensor([N for _ in range(B)]).cumsum(0)

        model_input = {
            'coord': pts.reshape(-1, 3),
            'grid_size': torch.ones([3]) * 1.0 / self.grid_resolution,
            'offset': offset.to(pts.device),
            'feat': feats,
        }
        model_input['grid_coord'] = torch.floor(model_input['coord'] * self.grid_resolution).int()

        # 2. predict delta parameters
        feat_delta = self.point_transformer(model_input)['feat']
        return feat_delta