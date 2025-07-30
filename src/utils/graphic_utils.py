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
import math
import numpy as np
from typing import NamedTuple
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.renderer import (
    PointsRasterizationSettings, 
    PointsRasterizer)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

from smplx import SMPLX

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View_torch(R, t):
    Rt = torch.zeros(4, 4, device=R.device)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

def getWorld2View2_torch(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros(4, 4, device=R.device)
    Rt[:3, :3] = R.transpose(0, 1) 
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate.to(R.device)) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix_torch(znear, zfar, fovX, fovY, K, w, h):
    # tanHalfFovY = math.tan((fovY / 2))
    # tanHalfFovX = math.tan((fovX / 2))

    # top = tanHalfFovY * znear
    # bottom = -top
    # right = tanHalfFovX * znear
    # left = -right

    # P = torch.zeros(4, 4).to(fovX.device)

    # z_sign = 1.0

    # P[0, 0] = 2.0 * znear / (right - left)
    # P[1, 1] = 2.0 * znear / (top - bottom)
    # P[0, 2] = (right + left) / (right - left)
    # P[1, 2] = (top + bottom) / (top - bottom)
    # P[3, 2] = z_sign
    # P[2, 2] = z_sign * zfar / (zfar - znear)
    # P[2, 3] = -(zfar * znear) / (zfar - znear)

    focalx, focaly = K[0, 0].item(), K[1, 1].item()
    px, py = K[0, 2].item(), K[1, 2].item()
    tanfovx = math.tan(fovX * 0.5)
    tanfovy = math.tan(fovY * 0.5)

    K_ndc = torch.tensor([
        [2 * focalx / w, 0, (2 * px - w) / w, 0],
        [0, 2 * focaly / h, (2 * py - h) / h, 0],
        [0, 0, zfar / (zfar - znear), -zfar * znear / (zfar - znear)],
        [0, 0, 1, 0]
    ]).float().to(K.device)

    return K_ndc

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def focal2fov_torch(focal, pixels):
    return 2*torch.atan(pixels/(2*focal))

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

def compute_face_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return face_normals

def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  # will have artifacts without negation

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale

def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals

def project_gaussians(gaussians, intrinsic, extrinsic):
    '''
    Project gaussians to image plane
    xyz: [B, N, 3]
    intrinsic: [3, 3]
    extrinsic: [4, 4]
    return: [B, N, 2]
    '''

    xyz = gaussians['xyz']
    rot = gaussians['rot']
    rot_mat = quaternion_to_matrix(rot)
    
    # Convert gaussian points from world coordinates to camera coordinates
    homogen_coord = torch.ones([xyz.shape[0], xyz.shape[1], 1], device=xyz.device)
    homogeneous_xyz = torch.cat([xyz, homogen_coord], dim=-1)  # [B, N, 4]
    
    # Apply extrinsic matrix

    cam_xyz = torch.matmul(extrinsic, homogeneous_xyz.transpose(-1, -2))  # [B, 4, N]
    cam_xyz = cam_xyz.transpose(-1, -2)[..., :3]  # [B, N, 3]

    cam_rot_mat = torch.matmul(extrinsic[:, :3, :3], rot_mat) # [N, 3, 3]
    cam_rot = matrix_to_quaternion(cam_rot_mat) # [N, 4]
    gaussians['rot'] = cam_rot
    
    # Apply intrinsic matrix
    projected_xy = torch.matmul(intrinsic, cam_xyz.transpose(-1, -2))  # [B, 3, N]
    projected_xy = projected_xy.transpose(-1, -2)
    projected_gaussians = projected_xy[..., :2] / projected_xy[..., 2:3]

    # gaussians['xyz'] = projected_gaussians
    return projected_gaussians

def project_xyz(xyz, intrinsic, extrinsic):
    '''
    Project xyz to image plane
    xyz: [B, N, 3]
    intrinsic: [B, 3, 3]
    extrinsic: [B, 4, 4]
    return: [B, N, 2]
    '''
    if xyz.ndim == 4:
        B, T, N, _ = xyz.shape
        xyz = xyz.reshape(B*T, N, 3)
        intrinsic = intrinsic.reshape(B*T, 3, 3)
        extrinsic = extrinsic.reshape(B*T, 4, 4)

    homogen_coord = torch.ones([xyz.shape[0], xyz.shape[1], 1], device=xyz.device)
    homogeneous_xyz = torch.cat([xyz, homogen_coord], dim=-1)  # [B, N, 4]

    cam_xyz = torch.matmul(extrinsic, homogeneous_xyz.transpose(-1, -2))  # [B, 4, N]
    cam_xyz = cam_xyz.transpose(-1, -2)[..., :3]  # [B, N, 3]

    projected_xy = torch.matmul(intrinsic, cam_xyz.transpose(-1, -2))  # [B, 3, N]
    projected_xy = projected_xy.transpose(-1, -2)
    projected_uv = projected_xy[..., :2] / projected_xy[..., 2:3]

    return projected_uv

def points_projection(points: torch.Tensor,
                    c2ws: torch.Tensor,
                    intrinsics: torch.Tensor,
                    local_features: torch.Tensor,
                    # Rasterization settings
                    raster_point_radius: float = 0.0075,  # point size
                    raster_points_per_pixel: int = 1,  # a single point per pixel, for now
                    bin_size: int = 0):
    '''
    Project points to image plane
    points: [B, Np, 3]
    c2ws: [B, 4, 4]
    intrinsics: [B, 3, 3]
    local_features: [B, C, H, W]
    '''

    B, C, H, W = local_features.shape
    device = local_features.device
    raster_settings = PointsRasterizationSettings(
            image_size=(H, W),
            radius=raster_point_radius,
            points_per_pixel=raster_points_per_pixel,
            bin_size=bin_size,
        )
    Np = points.shape[1]
    R = raster_settings.points_per_pixel

    # w2cs = torch.inverse(c2ws)
    w2cs = c2ws # test
    image_size = torch.as_tensor([H, W]).view(1, 2).expand(w2cs.shape[0], -1).to(device)
    cameras = cameras_from_opencv_projection(w2cs[:, :3, :3], w2cs[:, :3, 3], intrinsics, image_size)

    rasterize = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterize(Pointclouds(points))
    fragments_idx = fragments.idx.long()
    visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
    points_to_visible_pixels = fragments_idx[visible_pixels]

    # Reshape local features to (B, H, W, R, C)
    local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)

    # Get local features corresponding to visible points
    local_features_proj = torch.zeros(B * Np, C, device=device)
    local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
    local_features_proj = local_features_proj.reshape(B, Np, C)

    # visible_mask = (fragments_idx[..., 0] > -1).to(torch.uint8)  # [B, H, W]

    # plt.figure(figsize=(6, 6))
    # plt.imshow(visible_mask[0].detach().cpu().numpy(), cmap='gray')
    # plt.title("Visible Pixel Mask (1=visible, 0=hidden)")
    # plt.axis('off')
    # plt.savefig("visible_mask.png", bbox_inches='tight')
    # print(f"[✓] Saved mask visualization to: visible_mask.png")
    # plt.close()

    return local_features_proj
    

################THUMAN################
def _subdivide(vertices,
               faces,
               face_index=None,
               vertex_attributes=None,
               return_index=False):
    """
    this function is adapted from trimesh
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
    vertex_attributes : dict
      Contains (n, d) attribute data
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    index_dict : dict
      Only returned if `return_index`, {index of
      original face : index of new faces}.
    """
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_mask]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # the new faces_subset with correct winding
    f = np.column_stack([faces_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    # turn the mask back into integer indexes
    nonzero = np.nonzero(face_mask)[0]
    # new faces start past the original faces
    # but we've removed all the faces in face_mask
    start = len(faces) - len(nonzero)
    # indexes are just offset from start
    stack = np.arange(
        start, start + len(f) * 4).reshape((-1, 4))
    # reformat into a slightly silly dict for some reason
    index_dict = {k: v for k, v in zip(nonzero, stack)}

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            attr_tris = values[faces_subset]
            if key == 'so3':
                attr_mid = np.zeros([unique.shape[0], 3], values.dtype)
            elif key == 'scale':
                edge_len = np.linalg.norm(values[edges[unique][:, 1]] - values[edges[unique][:, 0]], axis=-1)
                attr_mid = np.ones([unique.shape[0], 3], values.dtype) * edge_len[..., None]
            else:
                attr_mid = values[edges[unique]].mean(axis=1)
            new_attributes[key] = np.vstack((
                values, attr_mid))
        return new_vertices, new_faces, new_attributes, index_dict

    if return_index:
        # turn the mask back into integer indexes
        nonzero = np.nonzero(face_mask)[0]
        # new faces start past the original faces
        # but we've removed all the faces in face_mask
        start = len(faces) - len(nonzero)
        # indexes are just offset from start
        stack = np.arange(
            start, start + len(f) * 4).reshape((-1, 4))
        # reformat into a slightly silly dict for some reason
        index_dict = {k: v for k, v in zip(nonzero, stack)}

        return new_vertices, new_faces, index_dict

    return new_vertices, new_faces


def subdivide(vertices, faces, attributes, return_edges=False):
    mesh = trimesh.Trimesh(vertices, faces, vertex_attributes=attributes)
    new_vertices, new_faces, new_attributes, index_dict = _subdivide(mesh.vertices, mesh.faces, vertex_attributes=mesh.vertex_attributes)
    if return_edges:
        edges = trimesh.Trimesh(new_vertices, new_faces).edges
        return new_vertices, new_faces, new_attributes, edges, index_dict
    else:
        return new_vertices, new_faces, new_attributes, index_dict


def clip_T_world(xyzs_world, K, E, H, W):
    xyzs = torch.cat([xyzs_world, torch.ones_like(xyzs_world[..., 0:1, :])], dim=-2)
    K_expand = torch.zeros_like(E)
    fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    K_expand[:, 0, 0] = 2.0 * fx / W
    K_expand[:, 1, 1] = 2.0 * fy / H
    K_expand[:, 0, 2] = 1.0 - 2.0 * cx / W
    K_expand[:, 1, 2] = 1.0 - 2.0 * cy / H
    znear, zfar = 1e-3, 1e3
    K_expand[:, 2, 2] = -(zfar + znear) / (zfar - znear)
    K_expand[:, 3, 2] = -1.
    K_expand[:, 2, 3] = -2.0 * zfar * znear / (zfar - znear)

    # gl_transform = torch.tensor([[1., 0, 0, 0],
    #                              [0, -1., 0, 0],
    #                              [0, 0, -1., 0],
    #                              [0, 0, 0, 1.]], device=K.device)
    # gl_transform = torch.eye(4, dtype=K.dtype, device=K.device)
    # gl_transform[1, 1] = gl_transform[2, 2] = -1.

    gl_transform = torch.tensor([[1., 0, 0, 0],
                             [0, -1., 0, 0],
                             [0, 0, -1., 0],
                             [0, 0, 0, 1.]], device='cuda')
    
    return (K_expand @ gl_transform[None] @ E @ xyzs).permute(0, 2, 1)


import os
import numpy as np
import torch
from smplx import SMPLX
import trimesh
import pyrender
from pyrender import Scene, Mesh, IntrinsicsCamera, DirectionalLight, PointLight, MetallicRoughnessMaterial, RenderFlags


# Ensure OSMesa/EGL context if needed
os.environ['PYOPENGL_PLATFORM'] = 'egl'

class SimpleMeshRenderer:
    """
    Offscreen mesh renderer using Pyrender.
    """
    def __init__(self, faces: np.ndarray, H: int, W: int):
        self.faces = faces
        self.width, self.height = W, H
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.width,
            viewport_height=self.height
        )

    def render_mesh(
        self,
        vertices: torch.Tensor,
        image: np.ndarray,
        K: torch.Tensor,
        E: torch.Tensor,
        base_color: tuple = (0.8, 0.8, 0.8),
        metallic: float = 0.0,
        roughness: float = 1.0,
        emissive: tuple = None
    ) -> np.ndarray:
        """
        Render a single SMPL-X mesh on a single image with its camera params.

        Args:
            vertices: (V,3) tensor of mesh vertices
            image:    (H,W,3) uint8 background image
            K:        (3,3) camera intrinsics tensor
            E:        (4,4) camera extrinsics tensor (world-to-camera)
        Returns:
            (H,W,3) uint8 rendered image
        """
        # Convert inputs
        verts = vertices.detach().cpu().numpy()
        img = image.astype(np.float32)
        cam_K = K.detach().cpu().numpy()
        cam_pose = np.linalg.inv(E.detach().cpu().numpy())
        cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]

        # Build mesh and scene
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        mat_kwargs = {
            "baseColorFactor": (base_color[0], base_color[1], base_color[2], 1.0),
            "metallicFactor": metallic,
            "roughnessFactor": roughness
        }
        if emissive is not None:
            mat_kwargs["emissiveFactor"] = (emissive[0], emissive[1], emissive[2])
        material = MetallicRoughnessMaterial(**mat_kwargs)

        mesh_pyr = Mesh.from_trimesh(mesh, material=material, smooth=False)

        scene = Scene()
        scene.add(mesh_pyr)

        fx, fy = float(cam_K[0,0]), float(cam_K[1,1])
        cx, cy = float(cam_K[0,2]), float(cam_K[1,2])

        camera = IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=cam_pose)

        key_light = DirectionalLight(color=[1.0,1.0,1.0], intensity=5.0)
        scene.add(key_light, pose=cam_pose)

        # Render
        color, depth = self.renderer.render(scene)
        
        # print(f"color min: {color.min()}, color max: {color.max()}")

        mask = (depth > 0)[..., None]
        out = img * (1 - mask) + color.astype(np.float32) * mask
        return out / 255.0


def init_smplx_model(
    device: torch.device,
    model_folder: str = '/home/liubingqi/work/data/SMPL_SMPLX/models_smplx_v1_1/models/smplx',
    num_expression_coeffs: int = 100,
    num_betas: int = 10,
    num_pca_comps: int = 12,
    flat_hand_mean: bool = True,
    use_pca: bool = False,
    create_body_pose: bool = False,
    create_betas: bool = False,
    create_global_orient: bool = False,
) -> SMPLX:
    """
    Initialize the SMPL-X model for inference.
    """
    model = SMPLX(
        model_folder,
        gender='neutral',
        create_body_pose=create_body_pose,
        create_betas=create_betas,
        create_global_orient=create_global_orient,
        create_transl=False,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_right_hand_pose=False,
        create_left_hand_pose=False,
        use_pca=use_pca,
        num_pca_comps=num_pca_comps,
        num_betas=num_betas,
        flat_hand_mean=flat_hand_mean,
        num_expression_coeffs=num_expression_coeffs
    ).to(device)
    model.eval()
    return model


def draw_smplx_on_image(
    images: np.ndarray,
    smplx_params: dict,
    Ks: torch.Tensor,
    Es: torch.Tensor,
    smplx_model: SMPLX,
    renderer: SimpleMeshRenderer
) -> np.ndarray:
    """
    Render SMPL-X meshes on a batch of images.

    Args:
        images:         (B, T, 3, H, W) uint8 array
        smplx_params:  dict of tensors each shape (B, T, dim)
        Ks:             (B, T, 3, 3) intrinsics per image
        Es:             (B, T, 4, 4) extrinsics per image
        smplx_model:    SMPLX model in eval mode
        renderer:       SimpleMeshRenderer instance

    Returns:
        results: (B, H, 2*W, 3) uint8 array of concatenated original and rendered
    """
    B, T, _, H, W  = images.shape
    images = images.reshape(B*T, 3, H, W)
    Ks = Ks.reshape(B*T, 3, 3)
    Es = Es.reshape(B*T, 4, 4)
    device = next(smplx_model.parameters()).device

    # Prepare SMPL-X inputs: each param (B, dim) -> model expects (B, dim)
    params = {}
    for key, tensor in smplx_params.items():
        params[key] = tensor.to(device).reshape(B*T, -1)

    # Forward SMPL-X
    with torch.no_grad():
        out = smplx_model(**params)
        verts_batch = out.vertices.detach()  # (B, V, 3)

    # print(f"verts_batch.shape: {verts_batch.shape}")

    results = []
    for b in range(B*T):
        img = images[b].permute(1, 2, 0).detach().cpu().numpy()
        K = Ks[b]
        E = Es[b]
        verts = verts_batch[b]

        # Render
        rendered = renderer.render_mesh(verts, img, K, E)

        # Concatenate
        combined = np.concatenate([img, rendered], axis=0)
        results.append(combined)

    return np.stack(results, axis=0)


### Gaussian Splatting Renderer ###

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

import os
import numpy as np
import torch
from smplx import SMPLX
import trimesh
import pyrender
from pyrender import Scene, Mesh, IntrinsicsCamera, DirectionalLight, PointLight, MetallicRoughnessMaterial, RenderFlags


# Ensure OSMesa/EGL context if needed
os.environ['PYOPENGL_PLATFORM'] = 'egl'

class SimpleMeshRenderer:
    """
    Offscreen mesh renderer using Pyrender.
    """
    def __init__(self, faces: np.ndarray, H: int, W: int):
        self.faces = faces
        self.width, self.height = W, H
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.width,
            viewport_height=self.height
        )

    def render_mesh(
        self,
        vertices: torch.Tensor,
        image: np.ndarray,
        K: torch.Tensor,
        E: torch.Tensor,
        base_color: tuple = (0.8, 0.8, 0.8),
        metallic: float = 0.0,
        roughness: float = 1.0,
        emissive: tuple = None
    ) -> np.ndarray:
        """
        Render a single SMPL-X mesh on a single image with its camera params.

        Args:
            vertices: (V,3) tensor of mesh vertices
            image:    (H,W,3) uint8 background image
            K:        (3,3) camera intrinsics tensor
            E:        (4,4) camera extrinsics tensor (world-to-camera)
        Returns:
            (H,W,3) uint8 rendered image
        """
        # Convert inputs
        verts = vertices.detach().cpu().numpy()
        img = image.astype(np.float32)
        cam_K = K.detach().cpu().numpy()
        cam_pose = np.linalg.inv(E.detach().cpu().numpy())
        cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]

        # Build mesh and scene
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        mat_kwargs = {
            "baseColorFactor": (base_color[0], base_color[1], base_color[2], 1.0),
            "metallicFactor": metallic,
            "roughnessFactor": roughness
        }
        if emissive is not None:
            mat_kwargs["emissiveFactor"] = (emissive[0], emissive[1], emissive[2])
        material = MetallicRoughnessMaterial(**mat_kwargs)

        mesh_pyr = Mesh.from_trimesh(mesh, material=material, smooth=False)

        scene = Scene()
        scene.add(mesh_pyr)

        fx, fy = float(cam_K[0,0]), float(cam_K[1,1])
        cx, cy = float(cam_K[0,2]), float(cam_K[1,2])
        # print(f"Camera intrinsics - fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        # print(f"Camera pose:\n{cam_pose}")
        # print(f"Vertices range - x: [{verts[:,0].min():.3f}, {verts[:,0].max():.3f}], "
        #       f"y: [{verts[:,1].min():.3f}, {verts[:,1].max():.3f}], "
        #       f"z: [{verts[:,2].min():.3f}, {verts[:,2].max():.3f}]")
        camera = IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=cam_pose)

        key_light = DirectionalLight(color=[1.0,1.0,1.0], intensity=5.0)
        scene.add(key_light, pose=cam_pose)

        # Render
        color, depth = self.renderer.render(scene)
        
        # print(f"color min: {color.min()}, color max: {color.max()}")

        mask = (depth > 0)[..., None]
        out = img * (1 - mask) + color.astype(np.float32) * mask
        return out / 255.0


def init_smplx_model(
    device: torch.device,
    model_folder: str = '/home/liubingqi/work/data/SMPL_SMPLX/models_smplx_v1_1/models/smplx'
) -> SMPLX:
    """
    Initialize the SMPL-X model for inference.
    """
    model = SMPLX(
        model_folder,
        gender='neutral',
        create_body_pose=False,
        create_betas=False,
        create_global_orient=False,
        create_transl=False,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_right_hand_pose=False,
        create_left_hand_pose=False,
        use_pca=False,
        num_pca_comps=12,
        num_betas=10,
        flat_hand_mean=True
    ).to(device)
    model.eval()
    return model


def draw_smplx_on_image(
    images: np.ndarray,
    smplx_params: dict,
    Ks: torch.Tensor,
    Es: torch.Tensor,
    smplx_model: SMPLX,
    renderer: SimpleMeshRenderer
) -> np.ndarray:
    """
    Render SMPL-X meshes on a batch of images.

    Args:
        images:         (B, T, 3, H, W) uint8 array
        smplx_params:  dict of tensors each shape (B, T, dim)
        Ks:             (B, T, 3, 3) intrinsics per image
        Es:             (B, T, 4, 4) extrinsics per image
        smplx_model:    SMPLX model in eval mode
        renderer:       SimpleMeshRenderer instance

    Returns:
        results: (B, H, 2*W, 3) uint8 array of concatenated original and rendered
    """
    B, T, _, H, W  = images.shape
    images = images.reshape(B*T, 3, H, W)
    Ks = Ks.reshape(B*T, 3, 3)
    Es = Es.reshape(B*T, 4, 4)
    device = next(smplx_model.parameters()).device

    # Prepare SMPL-X inputs: each param (B, dim) -> model expects (B, dim)
    params = {}
    for key, tensor in smplx_params.items():
        params[key] = tensor.to(device).reshape(B*T, -1)

    # Forward SMPL-X
    with torch.no_grad():
        out = smplx_model(**params)
        verts_batch = out.vertices.detach()  # (B, V, 3)

    results = []
    for b in range(B*T):
        img = images[b].permute(1, 2, 0).detach().cpu().numpy()
        K = Ks[b]
        E = Es[b]
        verts = verts_batch[b]

        # Render
        rendered = renderer.render_mesh(verts, img, K, E)

        # Concatenate
        combined = np.concatenate([img, rendered], axis=0)
        results.append(combined)

    return np.stack(results, axis=0)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_feature_maps(triplane: torch.Tensor, save_dir: str, save_name: str, batch_idx: int, num_channels: int = 3, normalize: bool = True, mode: str = 'pca'):
    """
    Visualize and save triplane slices.
    
    Args:
        triplane (Tensor): [B, 3, C, H, W] tensor
        num_channels (int): Number of channels per plane to visualize
        batch_idx (int): Batch index to visualize
        normalize (bool): Normalize each feature map to [0, 1]
        save_dir (str): Directory to save the output image
    """
    assert triplane.ndim == 5, "Expected shape [B, 3, C, H, W]"
    assert batch_idx < triplane.shape[0], f"Invalid batch_idx {batch_idx}"
    C = triplane.shape[2]
    num_channels = min(num_channels, C)

    os.makedirs(save_dir, exist_ok=True)
    planes = ['xy', 'xz', 'yz']
    triplane = triplane[batch_idx].detach().cpu()  # [3, C, H, W]

    fig, axs = plt.subplots(nrows=3, ncols=num_channels, figsize=(4 * num_channels, 10))

    if mode == 'channel':
        for plane_idx in range(3):
            for ch in range(num_channels):
                feat = triplane[plane_idx, ch]
                if normalize and feat.max() > feat.min():
                    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)

                ax = axs[plane_idx, ch] if num_channels > 1 else axs[plane_idx]
                im = ax.imshow(feat.numpy(), cmap='seismic', origin='lower')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{planes[plane_idx]} | Channel {ch}")

                # 可视化统计信息
                mean_val = feat.mean().item()
                ax.text(0.05, 0.95, f"mean={mean_val:.2f}", color='white',
                        fontsize=8, ha='left', va='top', transform=ax.transAxes)
    elif mode == "pca":
        triplane_sample = triplane
        for plane_idx in range(3):
            C, H, W = triplane_sample[plane_idx].shape
            feature_map = triplane_sample[plane_idx].reshape(C, -1).T

            pca = PCA(n_components=num_channels)
            reduced = pca.fit_transform(feature_map).reshape(H, W, num_channels)
            for ch in range(num_channels):
                feat = reduced[:, :, ch]
                if normalize and feat.max() > feat.min():
                    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)

                ax = axs[plane_idx, ch] if num_channels > 1 else axs[plane_idx]
                im = ax.imshow(feat, cmap='seismic', origin='lower')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{planes[plane_idx]} | Channel {ch}")
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{save_name}_batch{batch_idx}.png")
    plt.savefig(save_path)
    print(f"[✓] Saved triplane visualization to: {save_path}")
    plt.close()