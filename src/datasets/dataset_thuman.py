import os
import torch
import numpy as np
import json
import smplx
import cv2
import random

from torch.utils.data import Dataset

class ThumanDataset(Dataset):
    def __init__(self, 
                 dataset_root, 
                 smplx_params_path,
                 subj_list_path, 
                 image_size=(1024, 1024), 
                 device='cpu',
                 n_test=4,
                 pcd_nums=30000):
        self.dataset_root = dataset_root
        self.smplx_params_path = smplx_params_path
        with open(subj_list_path, 'r') as f:
            self.subj_list = [line.strip() for line in f.readlines()]
        self.image_size = image_size
        self.device = torch.device(device)
        self.n_test = n_test
        self.pcd_nums = pcd_nums
        self.data = []

        for subj in self.subj_list:
            subj_dir = os.path.join(self.dataset_root, subj, '12views_3')
            render_dir = os.path.join(subj_dir, 'render')
            calib_dir = os.path.join(subj_dir, 'calib')
            smplx_param_path = os.path.join(
                self.smplx_params_path,
                subj, 'smplx_param.pkl'
            )
            img_names = sorted([f for f in os.listdir(render_dir) if f.endswith('.png')])
            img_paths = [os.path.join(render_dir, img_name) for img_name in img_names]
            calib_paths = [os.path.join(calib_dir, img_name.replace('.png', '.json')) for img_name in img_names]
            self.data.append({
                'subj': subj,
                'img_paths': img_paths,
                'calib_paths': calib_paths,
                'smplx_param_path': smplx_param_path
            })

    def __len__(self):
        return len(self.data)

    def load_smplx_params(self, smplx_param_path):
        param = np.load(smplx_param_path, allow_pickle=True)
        param_dict = {}
        for key in param.keys():
            param_dict[key] = torch.as_tensor(param[key], dtype=torch.float32)

        param_dict['global_orient'] = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        param_dict['transl'] = torch.tensor([[0, 0.35, 0]], dtype=torch.float32)
        param_dict['body_pose'] = param_dict['body_pose'].reshape(1, -1, 3)
        param_dict['left_hand_pose'] = param_dict['left_hand_pose'].reshape(1, -1, 3)
        param_dict['right_hand_pose'] = param_dict['right_hand_pose'].reshape(1, -1, 3)
        
        return param_dict

    def construct_camera_params(self, calib_paths):
        intrinsic_list = []
        extrinsic_list = []
        fx, fy = 5000, 5000
        cx, cy = 512, 512
        for calib_path in calib_paths:
            with open(calib_path, 'r') as f:
                calib = json.load(f)
            c2w = torch.tensor(calib['cam_param'], dtype=torch.float32)
            extrinsic = torch.linalg.inv(c2w)
            intrinsic = torch.tensor([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]], dtype=torch.float32)
            intrinsic_list.append(intrinsic)
            extrinsic_list.append(extrinsic)
        intrinsic_stack = torch.stack(intrinsic_list, dim=0)  # (N, 3, 3)
        extrinsic_stack = torch.stack(extrinsic_list, dim=0)  # (N, 4, 4)
        return intrinsic_stack, extrinsic_stack
    
    def load_images(self, img_paths):

        imgs = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"no image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))  # (W, H)
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # (3, H, W)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)  # (N, 3, H, W)
        return imgs

    def __getitem__(self, idx):
        item = self.data[idx]
        img_paths = item['img_paths']
        calib_paths = item['calib_paths']
        smplx_param_path = item['smplx_param_path']
        smplx_params = self.load_smplx_params(smplx_param_path)

        total_imgs = len(img_paths)
        indices = [2]
        other_indices = list(range(total_imgs))
        other_indices.remove(2)
        random_indices = random.sample(other_indices, self.n_test - 1)
        indices.extend(random_indices)
        img_paths = [img_paths[i] for i in indices]
        calib_paths = [calib_paths[i] for i in indices]

        imgs = self.load_images(img_paths)
        intrinsic_stack, extrinsic_stack = self.construct_camera_params(calib_paths)

        pcd_path = os.path.join(self.dataset_root, item['subj'], '12views_3', 'vertices.npy')
        pcd_points = torch.from_numpy(np.load(pcd_path)).float()
        total_points = pcd_points.shape[0]
        if total_points >= self.pcd_nums:
            indices = np.linspace(0, total_points - 1, self.pcd_nums, dtype=int)
        else:
            indices = np.concatenate([
                np.arange(total_points),
                np.random.choice(total_points, self.pcd_nums - total_points, replace=True)
            ])
        pcd_points = pcd_points[indices]

        return {
            'imgs': imgs,
            'intrinsics': intrinsic_stack,
            'extrinsics': extrinsic_stack,
            'smplx_params': smplx_params,
            'subj': item['subj'],
            'pcd_points': pcd_points
        }

if __name__ == '__main__':
    dataset = ThumanDataset(
        dataset_root='/home/liubingqi/work/render/ICON/rendered_thuman',
        smplx_params_path='/disk2/liubingqi/data/THuman/thuman2.1_smplx/smplx/',
        subj_list_path='/disk2/liubingqi/data/rendered_thuman/train.txt',
        image_size=(1024, 1024),
        n_test=8,
    )
    print(len(dataset))
    sample = dataset[0]
    print("subject:", sample['subj'])
    print("imgs shape:", sample['imgs'].shape)
    print("intrinsics shape:", sample['intrinsics'].shape)
    print("extrinsics shape:", sample['extrinsics'].shape)
    print("smplx_params keys:", sample['smplx_params'].keys())
    for key in sample['smplx_params'].keys():
        print(key, sample['smplx_params'][key].shape)
    print("pcd_points shape:", sample['pcd_points'].shape)

    import smplx

    def project_gaussians(gaussians, intrinsic, extrinsic):
        xyz = gaussians
        homogen_coord = torch.ones([xyz.shape[0], xyz.shape[1], 1], device=xyz.device, dtype=torch.float32)
        homogeneous_xyz = torch.cat([xyz, homogen_coord], dim=-1)  # [B, N, 4]
        cam_xyz = torch.matmul(extrinsic, homogeneous_xyz.transpose(-1, -2))  # [B, 4, N]
        cam_xyz = cam_xyz.transpose(-1, -2)[..., :3]  # [B, N, 3]
        projected_xy = torch.matmul(intrinsic, cam_xyz.transpose(-1, -2))  # [B, 3, N]
        projected_xy = projected_xy.transpose(-1, -2)
        projected_gaussians = projected_xy[..., :2] / projected_xy[..., 2:3]
        return projected_gaussians

    subj = sample['subj']
    smplx_params = sample['smplx_params']
    device = torch.device("cpu")

    smpl_model = smplx.SMPLX(
        model_path="/home/liubingqi/work/data/SMPL_SMPLX/models_smplx_v1_1/models/smplx",
        create_global_orient=False,
        create_body_pose=False,
        create_betas=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_transl=False,
        use_pca=False
    ).to(device)

    param = {}
    for key in smplx_params.keys():
        param[key] = smplx_params[key].to(device)


    model_forward_params = dict(
        betas=param['betas'],
        global_orient=param['global_orient'],
        body_pose=param['body_pose'],
        left_hand_pose=param['left_hand_pose'],
        right_hand_pose=param['right_hand_pose'],
        jaw_pose=param['jaw_pose'],
        leye_pose=param['leye_pose'],
        reye_pose=param['reye_pose'],
        expression=param['expression'],
        transl=param['transl'],
        return_verts=True
    )

    smpl_out = smpl_model(**model_forward_params)

    intrinsic = sample['intrinsics'][0].to(device)
    extrinsic = sample['extrinsics'][0].to(device)

    verts = smpl_out.vertices # [1, N, 3]
    projected_verts = project_gaussians(verts, intrinsic, extrinsic).detach().cpu().numpy()

    print(sample['pcd_points'].shape)
    projected_pcd_points = project_gaussians(sample['pcd_points'].unsqueeze(0).detach().cpu(), intrinsic, extrinsic).detach().cpu().numpy()

    img = sample['imgs'][0]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    print("img shape:", img.shape)

    for vertex_idx, (x, y) in enumerate(projected_pcd_points[0, :, :2]):
        if 0 <= x < 1024 and 0 <= y < 1024:
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

    cv2.imwrite('projection.png', img)
    