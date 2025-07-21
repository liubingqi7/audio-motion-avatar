import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(eq=False)
class VideoData:
    """
    Dataclass for storing video chunks
    """

    video: torch.Tensor  # B, S, H, W, C
    smpl_parms: dict
    cam_parms: dict
    width: torch.Tensor
    height: torch.Tensor
    # optional data
    cropped_images: Optional[torch.Tensor] = None  # B, H, W, C
    segmentation: Optional[torch.Tensor] = None  # B, S, 1, H, W    
    depth_map: Optional[torch.Tensor] = None  # B, S, 1, H, W
    normal_map: Optional[torch.Tensor] = None  # B, S, 3, H, W
    sapiens_feats: Optional[torch.Tensor] = None  # B, S, 1536, 64, 64
    audio_features: Optional[torch.Tensor] = None  # B, S, 768
    pcd_points: Optional[torch.Tensor] = None  # B, N, 3

def cat_data(data1, data2):
    """
    Concatenate two VideoData objects along the batch dimension.
    """

    # cat dict
    smpl_parms = {k: torch.cat([data1.smpl_parms[k], data2.smpl_parms[k]], dim=0) 
                 for k in data1.smpl_parms.keys()}
    cam_parms = {k: torch.cat([data1.cam_parms[k], data2.cam_parms[k]], dim=0)
                for k in data1.cam_parms.keys()}
    assert data1.width == data2.width
    assert data1.height == data2.height

    return VideoData(
        video=torch.cat([data1.video, data2.video], dim=0),
        smpl_parms=smpl_parms,
        cam_parms=cam_parms,
        width=data1.width,
        height=data1.height,
    )



def collate_fn(batch):
    """
    Collate function for video data.
    """
    video = torch.stack([b.video for b in batch], dim=0)
    smpl_parms = {k: torch.stack([b.smpl_parms[k] for b in batch], dim=0) 
                  for k in batch[0].smpl_parms.keys()}
    cam_parms = {k: torch.stack([b.cam_parms[k] for b in batch], dim=0)
                 for k in batch[0].cam_parms.keys()}
    width = torch.stack([b.width for b in batch], dim=0)
    height = torch.stack([b.height for b in batch], dim=0)

    if batch[0].sapiens_feats is not None:
        feats = torch.stack([b.sapiens_feats for b in batch], dim=0)
        depth_map = torch.stack([b.depth_map for b in batch], dim=0)
        normal_map = torch.stack([b.normal_map for b in batch], dim=0)
    else:
        feats = None
        depth_map = None
        normal_map = None

    return VideoData(
        video=video,
        smpl_parms=smpl_parms,
        cam_parms=cam_parms,
        width=width,
        height=height,
        sapiens_feats=feats,
        depth_map=depth_map,
        normal_map=normal_map,
    )

def collate_fn_speech(batch):
    """
    Collate function for speech dataset.
    """
    batch_size = len(batch)
    clip_length = batch[0]['images'].shape[0]
    
    ref_idx = list(range(2))
    target_idx = list(range(clip_length-6, clip_length)) 

    cropped_images = torch.stack([b['cropped_images'][ref_idx] for b in batch], dim=0)
    
    ref_images = torch.stack([b['images'][ref_idx] for b in batch], dim=0)
    ref_smpl_params = {}
    for k in batch[0]['smplx_params'].keys():
        ref_smpl_params[k] = torch.stack([b['smplx_params'][k][ref_idx] for b in batch], dim=0)
    
    ref_cam_params = {}
    for k in batch[0]['camera'].keys():
        ref_cam_params[k] = torch.stack([b['camera'][k][ref_idx] for b in batch], dim=0)
    
    target_images = torch.stack([b['images'][target_idx] for b in batch], dim=0)
    target_smpl_params = {}
    for k in batch[0]['smplx_params'].keys():
        target_smpl_params[k] = torch.stack([b['smplx_params'][k][target_idx] for b in batch], dim=0)
    
    target_cam_params = {}
    for k in batch[0]['camera'].keys():
        target_cam_params[k] = torch.stack([b['camera'][k][target_idx] for b in batch], dim=0)

    audio_features = torch.stack([b['audio_feature'] for b in batch], dim=0)
    
    width = torch.ones(batch_size) * target_images.shape[-1]
    height = torch.ones(batch_size) * target_images.shape[-2]
    
    batch_train = VideoData(
        video=ref_images,
        smpl_parms=ref_smpl_params,
        cam_parms=ref_cam_params,
        width=width,
        height=height,
        cropped_images=cropped_images,
        sapiens_feats=None,
        depth_map=None,
        normal_map=None,
        audio_features=audio_features#[:, :2]
    )
    
    batch_test = VideoData(
        video=target_images,
        smpl_parms=target_smpl_params,
        cam_parms=target_cam_params,
        width=width,
        height=height,
        sapiens_feats=None,
        depth_map=None,
        normal_map=None,
        audio_features=audio_features#[:, 2:]
    )

    batch_id = batch[0]['batch_id']
    
    return batch_train, batch_test, batch_id

def collate_fn_idol_ori(batch):
    train_images = torch.stack([b['cond_imgs'][0:1] for b in batch], dim=0)

    train_smpl_params = {}
    for i in range(1):
        for k in batch[0]['cond_smpl_params'][i].keys():
            if k not in train_smpl_params:
                train_smpl_params[k] = []
            train_smpl_params[k].append(torch.stack([b['cond_smpl_params'][i][k] for b in batch], dim=0))
    train_smpl_params = {k: torch.stack(v, dim=1) for k, v in train_smpl_params.items()}

    train_cam_parms = {}
    train_cam_intrinsics = torch.stack([b['cond_intrinsics_matrix'][0:1] for b in batch], dim=0)
    train_cam_extrinsics = torch.stack([b['cond_poses'][0:1] for b in batch], dim=0)
    train_cam_parms = {
        'intrinsic': train_cam_intrinsics,
        'extrinsic': train_cam_extrinsics
    }

    test_images = torch.stack([b['cond_imgs'][1:] for b in batch], dim=0)
    test_smpl_params = {}
    for i in range(1, len(batch[0]['cond_smpl_params'])):
        for k in batch[0]['cond_smpl_params'][i].keys():
            if k not in test_smpl_params:
                test_smpl_params[k] = []
            test_smpl_params[k].append(torch.stack([b['cond_smpl_params'][i][k] for b in batch], dim=0))
    test_smpl_params = {k: torch.stack(v, dim=1) for k, v in test_smpl_params.items()}

    test_cam_parms = {}
    test_cam_intrinsics = torch.stack([b['cond_intrinsics_matrix'][1:] for b in batch], dim=0)
    test_cam_extrinsics = torch.stack([b['cond_poses'][1:] for b in batch], dim=0)
    test_cam_parms = {
        'intrinsic': test_cam_intrinsics,
        'extrinsic': test_cam_extrinsics
    }

    width = torch.ones(len(batch)) * train_images.shape[-1]
    height = torch.ones(len(batch)) * train_images.shape[-2]

    batch_train = VideoData(
        video=train_images,
        smpl_parms=train_smpl_params,
        cam_parms=train_cam_parms,
        width=width,
        height=height,
    )

    batch_test = VideoData(
        video=test_images,
        smpl_parms=test_smpl_params,
        cam_parms=test_cam_parms,
        width=width,
        height=height,
    )

    batch_id = 1

    return batch_train, batch_test, batch_id

def collate_fn_idol(batch):
    """
    Collate function for idol dataset.
    """
    time_dim = batch[0]['ref_images'].shape[0]    
    # Training data
    train_video = torch.stack([b['ref_images'] for b in batch], dim=0)
    train_smpl_parms = {}
    for i in range(len(batch[0]['ref_smpl_params'])):
        for k in batch[0]['ref_smpl_params'][i].keys():
            if k not in train_smpl_parms:
                train_smpl_parms[k] = []
            train_smpl_parms[k].append(torch.stack([b['ref_smpl_params'][i][k] for b in batch]))
    train_smpl_parms = {k: torch.stack(v, dim=1) for k, v in train_smpl_parms.items()}

    train_cam_parms = {}
    for i in range(len(batch[0]['ref_cam_params'])):
        for k in batch[0]['ref_cam_params'][i].keys():
            if k not in train_cam_parms:
                train_cam_parms[k] = []
            train_cam_parms[k].append(torch.stack([b['ref_cam_params'][i][k] for b in batch]))
    train_cam_parms = {k: torch.stack(v, dim=1) for k, v in train_cam_parms.items()}

    test_video = torch.stack([b['target_images'] for b in batch], dim=0)
    test_smpl_parms = {}
    for i in range(len(batch[0]['target_smpl_params'])):
        for k in batch[0]['target_smpl_params'][i].keys():
            if k not in test_smpl_parms:
                test_smpl_parms[k] = []
            test_smpl_parms[k].append(torch.stack([b['target_smpl_params'][i][k] for b in batch]))
    test_smpl_parms = {k: torch.stack(v, dim=1) for k, v in test_smpl_parms.items()}

    test_cam_parms = {}
    for i in range(len(batch[0]['target_cam_params'])):
        for k in batch[0]['target_cam_params'][i].keys():
            if k not in test_cam_parms:
                test_cam_parms[k] = []
            test_cam_parms[k].append(torch.stack([b['target_cam_params'][i][k] for b in batch]))
    test_cam_parms = {k: torch.stack(v, dim=1) for k, v in test_cam_parms.items()}

    # Fixed parameters
    width = torch.ones(len(batch)) * train_video.shape[-1]
    height = torch.ones(len(batch)) * train_video.shape[-2]

    batch_train = VideoData(
        video=train_video,
        smpl_parms=train_smpl_parms,
        cam_parms=train_cam_parms,
        width=width,
        height=height,
        sapiens_feats=None,
        depth_map=None,
        normal_map=None
    )
    
    batch_test = VideoData(
        video=test_video,
        smpl_parms=test_smpl_parms,
        cam_parms=test_cam_parms,
        width=width,
        height=height,
        sapiens_feats=None,
        depth_map=None,
        normal_map=None
    )

    return batch_train, batch_test

def collate_fn_thuman(batch):
    """
    Collate function for thuman dataset.
    """
    time_dim = batch[0].video.shape[0]
    # train_frames = time_dim - 1
    train_frames = 4
    
    train_video = torch.stack([b.video[:train_frames] for b in batch], dim=0)
    train_smpl_parms = {k: torch.stack([b.smpl_parms[k][:train_frames] for b in batch], dim=0) 
                      for k in batch[0].smpl_parms.keys()}
    train_cam_parms = {k: torch.stack([b.cam_parms[k][:train_frames] for b in batch], dim=0)
                     for k in batch[0].cam_parms.keys()}
    
    test_video = torch.stack([b.video[train_frames:] for b in batch], dim=0)
    test_smpl_parms = {k: torch.stack([b.smpl_parms[k][train_frames:] for b in batch], dim=0) 
                     for k in batch[0].smpl_parms.keys()}
    test_cam_parms = {k: torch.stack([b.cam_parms[k][train_frames:] for b in batch], dim=0)
                    for k in batch[0].cam_parms.keys()}
    
    if batch[0].sapiens_feats is not None:
        train_feats = torch.stack([b.sapiens_feats[:train_frames] for b in batch], dim=0)
        train_depth_map = torch.stack([b.depth_map[:train_frames] for b in batch], dim=0)
        train_normal_map = torch.stack([b.normal_map[:train_frames] for b in batch], dim=0)

        test_feats = torch.stack([b.sapiens_feats[train_frames:] for b in batch], dim=0)
        test_depth_map = torch.stack([b.depth_map[train_frames:] for b in batch], dim=0)
        test_normal_map = torch.stack([b.normal_map[train_frames:] for b in batch], dim=0)
    
    width = torch.stack([b.width for b in batch], dim=0)
    height = torch.stack([b.height for b in batch], dim=0)

    batch_train = VideoData(
        video=train_video,
        smpl_parms=train_smpl_parms,
        cam_parms=train_cam_parms,
        width=width,
        height=height,
        sapiens_feats=train_feats,
        depth_map=train_depth_map,
        normal_map=train_normal_map,
    )
    
    batch_test = VideoData(
        video=test_video,
        smpl_parms=test_smpl_parms,
        cam_parms=test_cam_parms,
        width=width,
        height=height,
        sapiens_feats=test_feats,
        depth_map=test_depth_map,
        normal_map=test_normal_map,
    )
    
    return (batch_train, batch_test)

def collate_fn_thuman_ori(batch):
    """
    针对ThumanOriDataset的collate函数，参考ThumanOriDataset和collate_fn_idol_ori实现。
    """

    cond_imgs = torch.stack([b['imgs'] for b in batch], dim=0)  # (B, n_test, 3, H, W)
    cond_intrinsics_matrix = torch.stack([b['intrinsics'] for b in batch], dim=0)  # (B, n_test, 3, 3)
    cond_poses = torch.stack([b['extrinsics'] for b in batch], dim=0)  # (B, n_test, 4, 4)

    n_test = cond_imgs.shape[1]
    cond_smpl_params = []
    for i in range(n_test):
        smpl_dict = {}
        for k in batch[0]['smplx_params'].keys():
            # (1, D) -> (B, 1, D) -> squeeze(1)
            smpl_dict[k] = torch.stack([b['smplx_params'][k].squeeze(0) for b in batch], dim=0)
        cond_smpl_params.append(smpl_dict)

    train_imgs = cond_imgs[:, 0]  # (B, 3, H, W)
    train_intrinsics = cond_intrinsics_matrix[:, 0]  # (B, 3, 3)
    train_extrinsics = cond_poses[:, 0]  # (B, 4, 4)
    train_smpl_params = cond_smpl_params[0]

    test_imgs = cond_imgs[:, 1:]  # (B, n_test-1, 3, H, W)
    test_intrinsics = cond_intrinsics_matrix[:, 1:]  # (B, n_test-1, 3, 3)
    test_extrinsics = cond_poses[:, 1:]  # (B, n_test-1, 4, 4)
    test_smpl_params = cond_smpl_params[1:]

    train_pcd_points = torch.stack([b['pcd_points'] for b in batch], dim=0)  # (B, N, 3)
    test_pcd_points = torch.stack([b['pcd_points'] for b in batch], dim=0)  # (B, N, 3)

    batch_train = VideoData(
        video=train_imgs.unsqueeze(1),  # (B, 1, 3, H, W)
        smpl_parms={k: v.unsqueeze(1) for k, v in train_smpl_params.items()},  # (B, 1, D)
        cam_parms={
            'intrinsic': train_intrinsics.unsqueeze(1),  # (B, 1, 3, 3)
            'extrinsic': train_extrinsics.unsqueeze(1),  # (B, 1, 4, 4)
        },
        width=torch.ones(len(batch)) * train_imgs.shape[-1],
        height=torch.ones(len(batch)) * train_imgs.shape[-2],
        pcd_points=train_pcd_points,
    )

    test_smpl_params_dict = {}
    for k in test_smpl_params[0].keys():
        test_smpl_params_dict[k] = torch.stack([d[k] for d in test_smpl_params], dim=1)  # (B, n_test-1, D)

    batch_test = VideoData(
        video=test_imgs,  # (B, n_test-1, 3, H, W)
        smpl_parms=test_smpl_params_dict,  # (B, n_test-1, D)
        cam_parms={
            'intrinsic': test_intrinsics,  # (B, n_test-1, 3, 3)
            'extrinsic': test_extrinsics,  # (B, n_test-1, 4, 4)
        },
        width=torch.ones(len(batch)) * test_imgs.shape[-1],
        height=torch.ones(len(batch)) * test_imgs.shape[-2],
        pcd_points=test_pcd_points,
    )

    batch_id = 1

    return batch_train, batch_test, batch_id

def collate_fn_zjumocap(batch):
    """
    Collate function for zjumocap dataset.
    """
    batch_train = [item['train'] for item in batch]
    batch_test = [item['test'] for item in batch]
    
    batch_train = collate_fn(batch_train)
    batch_test = collate_fn(batch_test)

    return (batch_train, batch_test)



def collate_fn_train(batch):
    """
    Collate function for video tracks data during training.
    """
    gotit = [gotit for _, gotit in batch]
    video = torch.stack([b.video for b, _ in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b, _ in batch], dim=0)
    visibility = torch.stack([b.visibility for b, _ in batch], dim=0)
    valid = torch.stack([b.valid for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    return (
        CoTrackerData(
            video=video,
            trajectory=trajectory,
            visibility=visibility,
            valid=valid,
            seq_name=seq_name,
        ),
        gotit,
    )


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj
