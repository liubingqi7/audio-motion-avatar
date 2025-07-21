import torch
import torch.nn as nn
import einops
from omegaconf import DictConfig
from src.models.transformers import Transformer1D_nn

class TriPlaneTemporalReducer(nn.Module):
    def __init__(self, C, time_steps):
        super().__init__()
        self.C = C
        self.T = time_steps
        self.planes = 3
        # depthwise 3D conv over time: each channel independently
        self.conv_time = nn.Conv3d(
            in_channels=self.planes * C,
            out_channels=self.planes * C,
            kernel_size=(self.T, 1, 1),
            stride=1,
            padding=(0, 0, 0),
            groups=self.planes * C,
            bias=False
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状 (B, T, 3, C, H, W)
        Returns:
            torch.Tensor: 输出形状 (B, 1, 3, C, H, W)
        """
        B, T, P, C, H, W = x.shape
        assert P == self.planes and T == self.T and C == self.C, \
            f"Expected (B,{self.T},3,{self.C},H,W), got {tuple(x.shape)}"
        # (B, T, 3, C, H, W) -> (B, 3*C, T, H, W)
        x_perm = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, P * C, T, H, W)
        # 3D conv over time -> (B, 3*C, 1, H, W)
        out = self.conv_time(x_perm)
        # 恢复 triplane 维度 -> (B, 3, C, 1, H, W)
        out = out.view(B, P, C, 1, H, W)
        # 调整维度顺序 -> (B, 1, 3, C, H, W)
        out = out.permute(0, 3, 1, 2, 4, 5).contiguous()
        return out

class SMPLXTemporalReducer(nn.Module):
    def __init__(self, C, time_steps):
        super().__init__()
        self.C = C 
        self.T = time_steps
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(C, C*2),
            nn.ReLU(),
            nn.Linear(C*2, C)
        )
        
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状 (B, T, C, S)
        Returns:
            torch.Tensor: 输出形状 (B, 1, C, S)
        """
        B, T, C, S = x.shape
        assert T == self.T and C == self.C, \
            f"Expected (B,{self.T},{self.C},S), got {tuple(x.shape)}"
            
        x = einops.rearrange(x, 'b t c s -> (b s) t c')
        
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        x = x.mean(dim=1, keepdim=True)  # (B*S, 1, C)
        
        out = einops.rearrange(x, '(b s) t c -> b t c s', b=B)
        
        return out

class AudioTriplaneNet(nn.Module):
    def __init__(self, cfg: DictConfig, renderer=None):
        super(AudioTriplaneNet, self).__init__()
        self.cfg = cfg.model.triplane_audio_net
        self.T_input = self.cfg.triplane_input_frames
        self.T_output = self.cfg.triplane_output_frames

        # self.triplane_downsample = nn.Sequential(
        #     nn.Conv2d(args.triplane_feature_dim, args.triplane_feature_dim, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(args.triplane_feature_dim, args.triplane_feature_dim, 3, stride=2, padding=1)
        # )
        
        # self.triplane_upsample = nn.Sequential(
        #     nn.ConvTranspose2d(args.triplane_feature_dim, args.triplane_feature_dim, 4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(args.triplane_feature_dim, args.triplane_feature_dim, 4, stride=2, padding=1)
        # )

        self.triplane_motion_encoder = TriPlaneTemporalReducer(
            C=self.cfg.triplane_feature_dim,
            time_steps=self.T_input
        )

        self.smplx_motion_encoder = SMPLXTemporalReducer(
            C=self.cfg.smpl_token_dim,
            time_steps=self.T_input
        )

        # self.triplane_tokenizer = TriplaneLearnablePositionalEmbedding(
        #     num_channels=args.triplane_feature_dim,
        #     plane_size=args.triplane_resolution
        # )

        self.triplane_token_len = 3*self.cfg.triplane_resolution*self.cfg.triplane_resolution
        self.smplx_token_len = self.cfg.smpl_token_len
        # self.triplane_tokenizer_high_res = TriplaneLearnablePositionalEmbedding(
        #     num_channels=args.triplane_feature_dim,
        #     plane_size=args.triplane_resolution  # After upsampling
        # )
                
        self.transformer = Transformer1D_nn(
            num_layers=self.cfg.transformer_layers,
            attention_head_dim=self.cfg.transformer_head_dim,
            in_channels=self.cfg.triplane_feature_dim,
            num_attention_heads=self.cfg.transformer_num_heads,
            cross_attention_dim=self.cfg.audio_feature_dim,
            norm_type="layer_norm",
            enable_memory_efficient_attention=False,
            gradient_checkpointing=True
        )
        
        # self.cross_attention = Transformer1D_nn(
        #     num_layers=4,
        #     attention_head_dim=64,
        #     in_channels=args.triplane_feature_dim,
        #     num_attention_heads=8,
        #     cross_attention_dim=1536,  # Image feature dim
        #     norm_type="layer_norm",
        #     enable_memory_efficient_attention=False,
        #     gradient_checkpointing=True
        # )

        # self.renderer = Renderer(args, smpl_decoder=smpl_decoder)
        self.renderer = renderer

    def forward(self, audio_features, input_triplane_tokens, ref_image_features, cam_params, smpl_tokens):
        """
        Args:
            audio_features: [B, T_audio, C_audio]  
            input_triplanes: [B, T_input, C, 3*H*W]
            ref_image_features: [B, C_img, H, W]
            smpl_tokens: []

        Returns:
            output_triplanes: [B, T_out, 3, C, H, W]
        """
        # print(f'audio_features.shape: {audio_features.shape}')
        # print(f'input_triplanes.shape: {input_triplanes.shape}')
        # print(f'smpl_tokens.shape: {smpl_tokens.shape}')
        # print(f'ref_image_features.shape: {ref_image_features.shape}')
        B, _, _ = audio_features.shape
        T_audio = audio_features.shape[1]

        # input_triplane_tokens = einops.rearrange(
        #     input_triplane_tokens,
        #     '(b t) c s -> b t c s',
        #     b=B,
        # )

        input_triplanes = einops.rearrange(
            input_triplane_tokens, 
            'b t c (np h w) -> b t np c h w', 
            np=3,
            h=self.cfg.triplane_resolution,
            w=self.cfg.triplane_resolution
        )
        
        input_triplane_motion = self.triplane_motion_encoder(input_triplanes).squeeze(1)
        # print(f'input_triplane_motion.shape: {input_triplane_motion.shape}')
        # input_triplane_motion_tokens = self.triplane_tokenizer(batch_size=B, cond_embeddings=input_triplane_motion)
        input_triplane_motion_tokens = einops.rearrange(
            input_triplane_motion,
            'b np c h w -> b c (np h w)',
        )
        input_smplx_motion_tokens = self.smplx_motion_encoder(smpl_tokens).squeeze(1)
        # last_triplane_tokens = self.triplane_tokenizer(batch_size=B, cond_embeddings=input_triplanes[:, -1])
        last_triplane_tokens = input_triplane_tokens[:, -1]
        last_smplx_tokens = smpl_tokens[:, -1]
        # print(f'input_triplane_motion_tokens.shape: {input_triplane_motion_tokens.shape}')
        # print(f'input_smplx_motion_tokens.shape: {input_smplx_motion_tokens.shape}')
        # print(f'last_triplane_tokens.shape: {last_triplane_tokens.shape}')
        
        init_query_tokens = torch.cat([input_triplane_motion_tokens, input_smplx_motion_tokens, last_triplane_tokens, last_smplx_tokens], dim=-1)

        output_triplanes = []
        output_triplane_tokens = []
        output_smpl_tokens = []
        query_tokens = init_query_tokens
        for t in range(self.T_output):
            output_tokens = self.transformer(query_tokens, audio_features[:, t:t+1])
            smpl_tokens = output_tokens[:, :, -self.smplx_token_len:]
            triplane_tokens = output_tokens[:, :, -self.triplane_token_len-self.smplx_token_len:-self.smplx_token_len]

            # pred_triplane = self.triplane_tokenizer.detokenize(triplane_tokens).unsqueeze(1)
            pred_triplane = einops.rearrange(
                triplane_tokens, 
                'b c (np h w) -> b np c h w', 
                np=3, 
                h=self.cfg.triplane_resolution, 
                w=self.cfg.triplane_resolution
            ).unsqueeze(1)
            
            last_triplane_tokens = output_triplanes[-1] if len(output_triplanes) > 0 else last_triplane_tokens
            # last_triplane = self.triplane_tokenizer.detokenize(last_triplane_tokens).unsqueeze(1)
            last_triplane = einops.rearrange(
                last_triplane_tokens,
                'b c (np h w) -> b np c h w', 
                np=3, 
                h=self.cfg.triplane_resolution, 
                w=self.cfg.triplane_resolution
            ).unsqueeze(1)

            last_smplx_tokens = output_smpl_tokens[-1] if len(output_smpl_tokens) > 0 else last_smplx_tokens

            # print(f'pred_triplane.shape: {pred_triplane.shape}')
            # print(f'last_triplane.shape: {last_triplane.shape}')
            # print(f'last_smplx_tokens.shape: {last_smplx_tokens.shape}')

            triplane_motion = self.triplane_motion_encoder(torch.cat([pred_triplane, last_triplane], dim=1)).squeeze(1)
            # triplane_motion_tokens = self.triplane_tokenizer(batch_size=B, cond_embeddings=triplane_motion)
            triplane_motion_tokens = einops.rearrange(
                triplane_motion,
                'b np c h w -> b c (np h w)',
            )
            smplx_motion_tokens = self.smplx_motion_encoder(torch.cat([last_smplx_tokens.unsqueeze(1), smpl_tokens.unsqueeze(1)], dim=1)).squeeze(1)
            
            # print(f'triplane_motion_tokens.shape: {triplane_motion_tokens.shape}')
            # print(f'smplx_motion_tokens.shape: {smplx_motion_tokens.shape}')
            # print(f'triplane_tokens.shape: {triplane_tokens.shape}')
            # print(f'smpl_tokens.shape: {smpl_tokens.shape}')
            
            query_tokens = torch.cat([triplane_motion_tokens, smplx_motion_tokens, triplane_tokens, smpl_tokens], dim=-1)
            
            output_triplanes.append(triplane_tokens)
            output_triplane_tokens.append(triplane_tokens)
            output_smpl_tokens.append(smpl_tokens)


        # output_triplanes = torch.stack(output_triplanes, dim=1)  # [B, T, C, S]
        output_triplane_tokens = torch.stack(output_triplane_tokens, dim=1)  # [B, T, C, S]
        # print(f'output_triplanes.shape: {output_triplanes.shape}')
        # output_triplanes = einops.rearrange(output_triplanes, 'b t c s -> (b t) c s')  # [B*T, C, S]
        # output_triplanes = self.triplane_tokenizer.detokenize(output_triplanes)  # [B*T, 3, C, H, W]
        # output_triplanes = einops.rearrange(output_triplanes, '(b t) np c h w -> b t np c h w', b=B)  # [B, T, 3, C, H, W]
        output_smpl_tokens = torch.stack(output_smpl_tokens, dim=1)
        # print(f'output_smpl_tokens.shape: {output_smpl_tokens.shape}')

        rendered_images, gaussians, smpl_params = self.renderer(output_triplane_tokens, cam_params, output_smpl_tokens)
        
        return rendered_images, gaussians, smpl_params, output_triplane_tokens, output_smpl_tokens