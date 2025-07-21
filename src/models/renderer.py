import torch
import torch.nn.functional as F
import torch.nn as nn
from omegaconf import DictConfig
from smplx import SMPLX
from src.models.point_transformer.point_encoder import PTv3Encoder
import einops
from src.utils.math_utils import inverse_sigmoid


class Renderer(nn.Module):
    def __init__(self, cfg: DictConfig = None, smpl_decoder=None):
        super(Renderer, self).__init__()
        self.cfg = cfg
        self.smplx_model = self.init_smplx_model()

        if cfg.predict_smplx_params:
            self.smpl_decoder = smpl_decoder
        
        if not cfg.no_point_refiner:
            self.point_encoder = PTv3Encoder(
                cfg=cfg
            )
            
            self.point_refiner = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3)
            )
            nn.init.constant_(self.point_refiner[-1].weight, 0)
            nn.init.constant_(self.point_refiner[-1].bias, 0)
            
        self.gaussian_decoder = nn.Module()
        
        self.gaussian_decoder.xyz_layer = nn.Linear(cfg.triplane_feature_dim*3 + 3, 3)
        self.gaussian_decoder.rotation_layer = nn.Linear(cfg.triplane_feature_dim*3 + 3, 4)
        self.gaussian_decoder.scaling_layer = nn.Linear(cfg.triplane_feature_dim*3 + 3, 3)
        self.gaussian_decoder.opacity_layer = nn.Linear(cfg.triplane_feature_dim*3 + 3, 1)
        self.gaussian_decoder.shs_layer = nn.Linear(cfg.triplane_feature_dim*3 + 3, 3)
        
        nn.init.constant_(self.gaussian_decoder.xyz_layer.weight, 0)
        nn.init.constant_(self.gaussian_decoder.xyz_layer.bias, 0)
        
        nn.init.constant_(self.gaussian_decoder.rotation_layer.weight, 0)
        nn.init.constant_(self.gaussian_decoder.rotation_layer.bias, 0)
        nn.init.constant_(self.gaussian_decoder.rotation_layer.bias[0], 1.0)
        
        nn.init.constant_(self.gaussian_decoder.scaling_layer.weight, 0)
        nn.init.constant_(self.gaussian_decoder.scaling_layer.bias, -1.0)
        
        nn.init.constant_(self.gaussian_decoder.opacity_layer.weight, 0)
        nn.init.constant_(self.gaussian_decoder.opacity_layer.bias, inverse_sigmoid(0.1))
        
        nn.init.constant_(self.gaussian_decoder.shs_layer.weight, 0)
        nn.init.constant_(self.gaussian_decoder.shs_layer.bias, 0)
        
    def forward(self, triplane_features, cam_params, smpl_tokens=None, smpl_params_gt=None):
        B, T, _, _ = smpl_tokens.shape
        triplane_features = einops.rearrange(
            triplane_features,
            "B T Ct (Np Hp Wp) -> (B T) Np Ct Hp Wp",
            Np=3,
            Hp=self.cfg.triplane_resolution,
            Wp=self.cfg.triplane_resolution,
        )

        smpl_tokens = einops.rearrange(smpl_tokens, 'b t c s -> (b t) c s')
        if self.smpl_decoder is not None:
            pred_smpl_params = self.smpl_decoder(smpl_tokens)
            for key in pred_smpl_params.keys():
                if key in ['betas', 'transl', 'global_orient', 'expression', 'jaw_pose', 'leye_pose', 'reye_pose']:
                    pred_smpl_params[key] = pred_smpl_params[key].reshape(B, T, -1)
                elif key in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
                    orig_shape = pred_smpl_params[key].shape
                    pred_smpl_params[key] = pred_smpl_params[key].reshape(B, T, *orig_shape[1:])
                elif key in ['focal']:
                    pred_smpl_params[key] = pred_smpl_params[key].reshape(B, T, -1)

        if smpl_params_gt is not None:
            smpl_params = smpl_params_gt
        else:
            smpl_params = pred_smpl_params

        initial_points = self.get_smpl_vertices(smpl_params)  # [B, N, 3]
        B, N, _ = initial_points.shape
        
        
        initial_features = self.sample_from_triplane(triplane_features, initial_points)  # [B, N, C]
        point_features = self.point_encoder(initial_points, initial_features)  # [B, N, 256]
        point_offsets = self.point_refiner(point_features).reshape(B, N, 3)  # [B, N, 3]
        refined_points = initial_points + point_offsets  # [B, N, 3]
        
        refined_features = self.sample_from_triplane(triplane_features, refined_points)  # [B, N, C]
        
        decoder_input = torch.cat([refined_points, refined_features], dim=-1)  # [B, N, 3+C]
        
        xyz_offset = self.gaussian_decoder.xyz_layer(decoder_input)  # [B, N, 3]
        rotation = self.gaussian_decoder.rotation_layer(decoder_input)  # [B, N, 4]
        scaling = self.gaussian_decoder.scaling_layer(decoder_input)  # [B, N, 3]
        opacity = self.gaussian_decoder.opacity_layer(decoder_input)  # [B, N, 1]
        shs = self.gaussian_decoder.shs_layer(decoder_input)  # [B, N, 3]
        
        gaussian_params = {
            'xyz_offset': xyz_offset,
            'rotation': rotation,
            'scaling': scaling,
            'opacity': opacity,
            'shs': shs
        }
        
        gaussians = self.construct_gaussians(gaussian_params, refined_points, smpl_params)
        
        rendered_images = render_batch(gaussians, cam_params['intrinsic'], cam_params['extrinsic'], self.cfg, debug=False)
        
        if self.cfg.predict_smplx_params:
            return rendered_images, gaussians, pred_smpl_params
        else:
            return rendered_images, gaussians
    
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
    
    def get_smpl_vertices(self, smpl_params):
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
            expression=expression,
            # transl=transl
        )
        
        vertices = output.vertices
        
        # B, N, _ = vertices.shape
        # faces = self.smplx_model.faces
        # face_verts = vertices[:, faces]  # [B, F, 3, 3]
        # face_centers = face_verts.mean(dim=2)  # [B, F, 3]
        # densified_verts = torch.cat([vertices, face_centers], dim=1)  # [B, N+F, 3]
        
        # vertices = densified_verts

        return vertices
    
    def sample_from_triplane(self, triplane_features, points):
        batched = points.ndim == 3
        if not batched:
            triplane_features = triplane_features[None, ...]
            points = points[None, ...]
        
        positions = torch.clamp(points / 2.0, -1, 1)
        
        indices2D = torch.stack(
            (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
            dim=-3,
        )
        
        out = F.grid_sample(
            einops.rearrange(triplane_features, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
            einops.rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        
        out = einops.rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
        
        if not batched:
            out = out.squeeze(0)
            
        return out
    
    def construct_gaussians(self, gaussian_params, points, smpl_params):
        delta_xyz = gaussian_params['xyz_offset']
        scale = gaussian_params['scaling']
        rotation = gaussian_params['rotation']
        opacity = gaussian_params['opacity']
        color = gaussian_params['shs']
        
        # max_step = 1.2 / 32000
        # delta_xyz = (torch.sigmoid(delta_xyz) - 0.5) * max_step
        
        # # 限制scale的大小，使其尽量小
        # scale = torch.clamp(scale, -5.0, 0.0)
        # scale = torch.exp(scale)
        
        rotation = F.normalize(rotation, dim=-1)
                
        color = torch.sigmoid(color)
        
        gaussians = {
            'xyz': points + delta_xyz + smpl_params['transl'].reshape(-1, 1, 3),
            'scale': scale,
            'rot': rotation,
            'opacity': opacity,
            'color': color,
            'shs': color,
        }
        
        return gaussians


### Gaussian Splatting Renderer ###

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import math
from src.utils.graphic_utils import getWorld2View2_torch, getProjectionMatrix_torch, focal2fov_torch
from src.utils.graphic_utils import eval_sh


SCALE_BIAS = 3.9
OPACITY_BIAS = 0.0

def render_multi_view(gaussians, K, E, args, bg_color=None, debug=False):
    '''
    Multi-view rendering for gaussian avatar
    gaussians: B, N, D
    K: B, T, 3, 3
    E: B, T, 4, 4
    '''
    B, T = E.shape[0], E.shape[1]
    
    expanded_gaussians = {k: v.unsqueeze(1).expand(-1, T, -1, -1) for k, v in gaussians.items()}
    expanded_gaussians = {k: v.reshape(B*T, -1, v.shape[-1]) for k, v in expanded_gaussians.items()}
    
    rendered_image = render_batch(expanded_gaussians, K, E, args, bg_color, debug)
    
    return rendered_image.reshape(B, T, args.image_size[0], args.image_size[1], 3)

def render_batch(gaussians, K, E, args, bg_color=None, debug=False):
    '''
    Batch rendering for gaussian avatar
    Args:
        gaussians: dict containing gaussian parameters with batch dimension
        K: intrinsic matrix [B, T, 3, 3]
        E: extrinsic matrix [B, T, 4, 4] 
        args: configuration arguments
        bg_color: background color, default white
        debug: whether to print debug info
    Returns:
        rendered: rendered images [B*T, H, W, 3]
    '''
    B, T = E.shape[0], E.shape[1]
    
    # Flatten batch and time dimensions
    E_flat = E.reshape(-1, 4, 4).float()
    K_flat = K.reshape(-1, 3, 3).float()

    # Split gaussians into different dict
    xyzs = gaussians['xyz'].reshape(B*T, -1, 3).float()
    rots = gaussians['rot'].reshape(B*T, -1, 4).float()
    scales = gaussians['scale'].reshape(B*T, -1, 3).float()
    opacities = gaussians['opacity'].reshape(B*T, -1, 1).float()
    colors = gaussians['color'].reshape(B*T, -1, 3).float()

    # Render using base function
    rendered_images = []
    for i in range(B*T):
        rendered = render_one(xyzs[i], rots[i], scales[i], opacities[i], colors[i], K_flat[i], E_flat[i], args, bg_color, debug)
        rendered_images.append(rendered.permute(1, 2, 0))
    
    return torch.stack(rendered_images).reshape(B, T, args.image_size[0], args.image_size[1], 3)

def render_one(xyzs, rots, scales, opacities, colors, K, E, args, bg_color=None, debug=False):
    '''
    Customized render for gaussian avatar
    '''

    extrinsics = E
    intrinsics = K

    R = extrinsics[:3, :3].reshape(3, 3).transpose(1, 0)
    T = extrinsics[:3, 3]

    znear = 0.01
    zfar = 100.0

    height = args.image_size[0]
    width = args.image_size[1]

    focal_length_y = intrinsics[1, 1]
    focal_length_x = intrinsics[0, 0]
    
    FovY = focal2fov_torch(focal_length_y, height)
    FovX = focal2fov_torch(focal_length_x, width)

    tanfovx = math.tan(FovX * 0.5)
    tanfovy = math.tan(FovY * 0.5)

    world_view_transform = getWorld2View2_torch(R, T).transpose(0, 1)
    projection_matrix = getProjectionMatrix_torch(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY, K=intrinsics, w=width, h=height).transpose(0,1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    if bg_color is None:
        bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=world_view_transform.cuda(),
        projmatrix=full_proj_transform.cuda(),
        sh_degree=3,
        campos=camera_center.cuda(),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    scales = torch.min(torch.exp(scales-SCALE_BIAS), torch.tensor(0.1, device=scales.device))
    opacities = torch.sigmoid(opacities-OPACITY_BIAS)

    if debug:
        scales = torch.ones_like(scales, device=scales.device) * 0.01
        opacities = torch.ones_like(opacities, device=opacities.device) * 0.1

    colors_precomp = None
    if not args.rgb:
        shs_view = colors.reshape(xyzs.shape[0], -1, 3).transpose(1, 2)
        dir_pp = (xyzs - camera_center.repeat(xyzs.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(args.sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = torch.clamp(colors, 0.0, 1.0)

    screenspace_points = torch.zeros_like(xyzs, dtype=xyzs.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, inv_depth = rasterizer(
    # rendered_image, radii = rasterizer(
        means3D = xyzs,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacities,
        scales = scales,
        rotations = rots,
        cov3D_precomp = None)
    
    rendered_image = rendered_image.clamp(0, 1)
    
    return rendered_image