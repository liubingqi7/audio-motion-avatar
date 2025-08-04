import os
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .dataset_thuman import ThumanDataset
from .dataset_idol import AvatarDataset
from .dataset_speech_vid import GaussianAudioDataset
from src.utils.data_utils import collate_fn_thuman_ori, collate_fn_speech, collate_fn_idol_ori

DATASET_COLLATE_FN_MAP = {
    "ThumanDataset": collate_fn_thuman_ori,
    "GaussianAudioDataset": collate_fn_speech,
    "IDOLDataset": collate_fn_idol_ori,
}


class DatasetFactory:
    """数据集工厂类，用于根据配置创建不同的数据集"""
    
    @staticmethod
    def create_dataset(cfg: DictConfig, split: str = "train"):
        """
        根据配置创建数据集
        
        Args:
            cfg: 配置对象
            split: 数据集分割 ("train", "val", "test")
            
        Returns:
            数据集对象
        """
        dataset_cfg = cfg.dataset
        dataset_type = dataset_cfg.type
        
        if dataset_type == "ThumanDataset":
            return DatasetFactory._create_thuman_dataset(cfg, split)
        elif dataset_type == "IDOLDataset":
            return DatasetFactory._create_idol_dataset(cfg, split)
        elif dataset_type == "GaussianAudioDataset":
            return DatasetFactory._create_speech_vid_dataset(cfg, split)
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    @staticmethod
    def _create_thuman_dataset(cfg: DictConfig, split: str):
        """创建THuman数据集"""
        dataset_cfg = cfg.dataset
        
        if split == "train":
            subj_list_path = dataset_cfg.train_list
        elif split == "val":
            subj_list_path = dataset_cfg.val_list
        elif split == "test":
            subj_list_path = dataset_cfg.test_list
        else:
            raise ValueError(f"不支持的split: {split}")
        
        return ThumanDataset(
            dataset_root=dataset_cfg.dataset_root,
            smplx_params_path=dataset_cfg.smplx_params_path,
            subj_list_path=subj_list_path,
            image_size=tuple(dataset_cfg.image_size),
            device='cpu',
            n_test=dataset_cfg.get("n_test", 4),
            pcd_nums=dataset_cfg.get("pcd_nums", 30000)
        )
    
    @staticmethod
    def _create_idol_dataset(cfg: DictConfig, split: str):
        """创建IDOL数据集"""
        dataset_cfg = cfg.dataset
        if split == "train":
            cache_path = dataset_cfg.cache_path_train
        elif split == "val":
            cache_path = dataset_cfg.cache_path_val
        elif split == "test":
            cache_path = dataset_cfg.cache_path_test
        else:
            raise ValueError(f"unsupported split: {split}")

        return AvatarDataset(
            data_prefix=dataset_cfg.data_prefix,
            cache_path=cache_path,
            img_res=dataset_cfg.image_size,
            specific_observation_num=dataset_cfg.specific_observation_num,
            first_is_front=dataset_cfg.first_is_front,
            better_range=dataset_cfg.better_range,
            if_include_video_ref_img=dataset_cfg.if_include_video_ref_img,
            prob_include_video_ref_img=dataset_cfg.prob_include_video_ref_img,
            allow_k_angles_near_the_front=dataset_cfg.allow_k_angles_near_the_front,
            test_mode=dataset_cfg.test_mode,
            load_test_data=dataset_cfg.load_test_data
        )
    
    @staticmethod
    def _create_speech_vid_dataset(cfg: DictConfig, split: str):
        """创建Speech Video数据集"""
        dataset_cfg = cfg.dataset
        
        return GaussianAudioDataset(
            root_dir=dataset_cfg.root_dir,
            audio_file=dataset_cfg.audio_file,
            wav2vec2_model_path=dataset_cfg.wav2vec2_model_path,
            clip_length=dataset_cfg.clip_length,
            sample_rate=dataset_cfg.sample_rate,
        )
    
    @staticmethod
    def create_dataloader(cfg: DictConfig, split: str = "train"):
        """
        创建数据加载器
        
        Args:
            cfg: 配置对象
            split: 数据集分割 ("train", "val", "test")
            
        Returns:
            数据加载器对象
        """
        dataset = DatasetFactory.create_dataset(cfg, split)
        
        if split == "train":
            batch_size = cfg.training.batch_size
            shuffle = True
        else:
            batch_size = 1
            shuffle = False
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            collate_fn=DATASET_COLLATE_FN_MAP[cfg.dataset.type]
        )