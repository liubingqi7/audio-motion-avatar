import torch.nn as nn
import torch
from torch_scatter import scatter_mean, scatter_max
from smplx import SMPLX
import einops
from omegaconf import DictConfig

from src.models.tokenizers import TriplaneLearnablePositionalEmbedding
from src.models.transformers import Transformer1D_nn
from src.utils.graphic_utils import points_projection
from src.models.smplx_decoder import SMPLXDecoder

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class SMPLXTriplaneEncoder(nn.Module):
    '''
    To encode a smpl-x mesh into a triplane
    '''
    def __init__(self, cfg: DictConfig, smpl_decoder=None):
        super(SMPLXTriplaneEncoder, self).__init__()
        self.cfg = cfg
        self.triplane_resolution = cfg.triplane_resolution
        self.feature_dim = cfg.triplane_feature_dim
        self.smplx_model = self.init_smplx_model()
        self.num_verts = self.smplx_model.lbs_weights.shape[0]

        self.fc_pos = nn.Linear(3 + cfg.triplane_feature_dim, 2 * cfg.triplane_feature_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * cfg.triplane_feature_dim, cfg.triplane_feature_dim) 
            for _ in range(3)
        ])
        self.fc_c = nn.Linear(cfg.triplane_feature_dim, cfg.triplane_feature_dim)

        if cfg.sample_feature:
            self.vertex_emb = nn.Embedding(self.num_verts, cfg.triplane_feature_dim//2)
        else:
            self.vertex_emb = nn.Embedding(self.num_verts, cfg.triplane_feature_dim)

        if cfg.predict_smplx_params:
            # smplx predictor
            self.smpl_token_len = cfg.smpl_token_len
            self.smpl_token_dim = cfg.smpl_token_dim

            self.smpl_tokens = nn.Parameter(
                torch.randn(self.smpl_token_dim, self.smpl_token_len)
            )

            self.cross_attn = Transformer1D_nn(
                num_layers=cfg.smplx_transformer_layers,
                attention_head_dim=cfg.smplx_transformer_head_dim,
                in_channels=self.smpl_token_dim,
                num_attention_heads=cfg.smplx_transformer_num_heads,
                cross_attention_dim=cfg.image_feature_dim,
                norm_type="layer_norm",
                enable_memory_efficient_attention=False,
                gradient_checkpointing=True
            )
                    
            self.smpl_decoder = smpl_decoder
        
        self.actvn = nn.ReLU()
        self.scatter = scatter_max
        
    def forward(self, cam_params, img_tokens, smpl_params_gt=None, img=None):           
        B, T, S, C = img_tokens.shape

        if self.cfg.predict_smplx_params:
            pred_smpl_params, smpl_tokens = self.smpl_predictor(img_tokens)

        if smpl_params_gt is not None:
            smpl_params = smpl_params_gt
        else:
            smpl_params = pred_smpl_params
        
        verts = self.get_smplx_verts(smpl_params)

        vertex_indices = torch.arange(self.num_verts, device=verts.device).unsqueeze(0).expand(verts.shape[0], -1)
        verts_emb = self.vertex_emb(vertex_indices)

        if self.cfg.sample_feature:
            verts_with_trans = verts.reshape(B*T, -1, 3) + smpl_params['transl']
            img_feature = img.reshape(B*T, *img.shape[2:])
            cam_extrinsic = cam_params['extrinsic'].reshape(B*T, 4, 4)
            cam_intrinsic = cam_params['intrinsic'].reshape(B*T, 3, 3)
            sampled_features = points_projection(verts_with_trans, cam_extrinsic, cam_intrinsic, img_feature)
            verts_feat = torch.cat([verts_emb, sampled_features], dim=-1)
        else:
            verts_feat = verts_emb

        net = self.fc_pos(torch.cat([verts, verts_feat], dim=-1))
        net = self.blocks[0](net)

        coord = {}
        index = {}
        position = torch.clamp(verts, -2.0 + 1e-6, 2.0 - 1e-6)
        coord["xy"] = position[..., [0, 1]]
        coord["xz"] = position[..., [0, 2]] 
        coord["yz"] = position[..., [1, 2]]

        for key in coord.keys():
            x = (coord[key] * self.triplane_resolution).long()
            index[key] = x[..., 0] + self.triplane_resolution * x[..., 1]
            index[key] = torch.clamp(index[key], 0, self.triplane_resolution * self.triplane_resolution - 1)
            index[key] = index[key][:, None, :]

        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        smplx_triplanes = torch.stack([
            self.generate_plane_features(index["xy"], c),
            self.generate_plane_features(index["xz"], c), 
            self.generate_plane_features(index["yz"], c)
        ], dim=1).view(B, T, 3, -1, self.triplane_resolution, self.triplane_resolution)

        return smplx_triplanes, smpl_tokens, pred_smpl_params
    
    def smpl_predictor(self, image_features):
        B, T, S, C = image_features.shape

        query_tokens = self.smpl_tokens.unsqueeze(0).repeat(B*T, 1, 1)
        tokens = self.cross_attn(query_tokens, image_features.reshape(B*T, S, C))
        
        smpl_params = self.smpl_decoder(tokens)

        for key in smpl_params.keys():
            if key in ['betas', 'transl', 'global_orient', 'expression', 'jaw_pose', 'leye_pose', 'reye_pose']:
                smpl_params[key] = smpl_params[key].reshape(B, T, -1)
            elif key in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
                orig_shape = smpl_params[key].shape
                smpl_params[key] = smpl_params[key].reshape(B, T, *orig_shape[1:])

        return smpl_params, tokens

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.shape[0], c.shape[2]
        keys = xy.keys()

        c_out = 0
        for key in keys:
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.triplane_resolution ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def generate_plane_features(self, index, c):
        fea_plane = c.new_zeros(index.shape[0], self.feature_dim, self.triplane_resolution ** 2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(index.shape[0], self.feature_dim, self.triplane_resolution, self.triplane_resolution)
        return fea_plane
    
    def init_smplx_model(self):
        """Initialize the SMPL-X model with predefined settings."""
        body_model = SMPLX(self.cfg.smplx_model_path,
                            gender="neutral", 
                            create_body_pose=False, 
                            create_betas=False, 
                            create_global_orient=False, 
                            create_transl=False,
                            create_expression=False,
                            create_jaw_pose=False, 
                            create_leye_pose=False, 
                            create_reye_pose=False, 
                            create_right_hand_pose=False,
                            create_left_hand_pose=False,
                            use_pca=False,
                            # num_pca_comps=12,
                            num_betas=10,
                            flat_hand_mean=self.cfg.flat_hand_mean,
                            num_expression_coeffs=self.cfg.num_expression_coeffs).to(self.cfg.device)
        return body_model

    def get_smplx_verts(self, smpl_params):
        B, T = smpl_params['global_orient'].shape[:2]
        
        # scale = smpl_params['scale'].reshape(B*T, -1)
        # transl = smpl_params['transl'].reshape(B*T, -1)
        global_orient = smpl_params['global_orient'].reshape(B*T, -1)
        body_pose = smpl_params['body_pose'].reshape(B*T, -1) 
        betas = smpl_params['betas'].reshape(B*T, -1)
        left_hand_pose = smpl_params['left_hand_pose'].reshape(B*T, -1)
        right_hand_pose = smpl_params['right_hand_pose'].reshape(B*T, -1)
        jaw_pose = smpl_params['jaw_pose'].reshape(B*T, -1)
        leye_pose = smpl_params['leye_pose'].reshape(B*T, -1)
        reye_pose = smpl_params['reye_pose'].reshape(B*T, -1)
        expression = smpl_params['expression'].reshape(B*T, -1)

        output = self.smplx_model(
            global_orient=global_orient,
            body_pose=body_pose, 
            betas=betas,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expression
        )
        
        vertices = output.vertices

        return vertices
    
    
class FeatureFusionNetwork(nn.Module):
    def __init__(self, cfg: DictConfig, feature_dim=64):
        super(FeatureFusionNetwork, self).__init__()
        self.cfg = cfg

        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_feature_dim = cfg.triplane_feature_dim
        self.triplane_tokenizer_geometry = TriplaneLearnablePositionalEmbedding(
            num_channels=self.triplane_feature_dim,
            plane_size=self.triplane_resolution
        )

        # self.triplane_tokenizer_image = TriplaneLearnablePositionalEmbedding(
        #     num_channels=self.triplane_feature_dim,
        #     plane_size=self.triplane_resolution
        # )

        self.transformer_cross = Transformer1D_nn(
            num_layers=cfg.cross_transformer_layers,
            attention_head_dim=cfg.cross_transformer_head_dim,
            in_channels=self.triplane_feature_dim,
            num_attention_heads=cfg.cross_transformer_num_heads,
            cross_attention_dim=1536,
            norm_type="layer_norm",
            enable_memory_efficient_attention=False,
            gradient_checkpointing=True
        )

        # self.transformer_self = Transformer1D_nn(
        #     num_layers=4,
        #     attention_head_dim=64,
        #     num_attention_heads=8,
        #     in_channels=self.triplane_feature_dim,
        #     enable_memory_efficient_attention=False,
        #     gradient_checkpointing=True
        # )

    def forward(self, geometry_triplane, image_features, smpl_tokens):        
        B, T, _, C, H, W = geometry_triplane.shape
        geometry_triplane = einops.rearrange(geometry_triplane, 'b t np c h w -> (b t) np c h w')
        image_features = einops.rearrange(image_features, 'b t S c -> (b t) S c')
        
        # tokenize geometry_triplane and image_triplane
        geometry_triplane_tokens = self.triplane_tokenizer_geometry(batch_size=B*T, cond_embeddings=geometry_triplane)
        # image_triplane_tokens = self.triplane_tokenizer_image(batch_size=B*T, cond_embeddings=image_features)
        # image_triplane_tokens = einops.rearrange(image_triplane_tokens, 'b c s -> b s c')
        # print(f"特征标记化后 - 显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        combined_tokens = torch.cat([geometry_triplane_tokens, smpl_tokens], dim=2)
        
        combined_tokens_out = self.transformer_cross(combined_tokens, image_features)
        
        tokens, smpl_tokens_out = torch.split(combined_tokens_out, 
                                             [geometry_triplane_tokens.shape[2], smpl_tokens.shape[2]], 
                                             dim=2)
        # print(f"第一次Transformer后 - 显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # cross view attention
        # tokens shape: [B*T, C, Seq_len]
        # tokens = einops.rearrange(tokens, '(b t) c s -> b c (t s)', b=B, t=T)
        # tokens = self.transformer_self(tokens)
        # tokens = einops.rearrange(tokens, 'b c (t s) -> (b t) c s', b=B, t=T)
        # print(f"第二次Transformer后 - 显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # # detokenize
        # fused_triplane = self.triplane_tokenizer_geometry.detokenize(tokens)
        # fused_triplane = einops.rearrange(fused_triplane, '(b t) np c h w -> b t np c h w', b=B, t=T)
        triplane_tokens = einops.rearrange(tokens, '(b t) c s-> b t c s', b=B, t=T)
        smpl_tokens = einops.rearrange(smpl_tokens_out, '(b t) c s -> b t c s', b=B, t=T)

        return triplane_tokens, smpl_tokens

