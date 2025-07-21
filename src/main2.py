import os
import torch
import lightning as L
from omegaconf import OmegaConf, DictConfig
import argparse
import torch.nn.functional as F
from src.utils.loss_utils import l1_loss, smplx_param_loss
import numpy as np
import matplotlib.pyplot as plt

from src.datasets.dataset_factory import DatasetFactory
from src.models.lightning_model_wrapper import AudioDrivenTriplaneAvatarLightning
from src.utils.trainer_factory import TrainerFactory
from src.configs.config_loader import ConfigLoader
from src.utils.graphic_utils import draw_smplx_on_image, init_smplx_model, SimpleMeshRenderer


def load_config(config_path: str) -> DictConfig:
    cfg = ConfigLoader.load_config(config_path)
    
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    return cfg

def setup_output_dirs(cfg: DictConfig):

    training_cfg = cfg.training if hasattr(cfg, 'training') else cfg
    output_dir = training_cfg.output_dir if hasattr(training_cfg, 'output_dir') else "./outputs_2"
    experiment_name = cfg.experiment_name if hasattr(cfg, 'experiment_name') else "experiment"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "target_images"), exist_ok=True)
    
    config_save_path = os.path.join(output_dir, f"{experiment_name}_config.yaml")
    OmegaConf.save(cfg, config_save_path)
    print(f"config saved to: {config_save_path}")

def main():
    parser = argparse.ArgumentParser(description="Audio Motion Avatar Training")
    parser.add_argument("--config", type=str, default="src/configs/config_ted_driven.yaml", 
                       help="config file path")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "test", "demo"],
                       help="run mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="checkpoint file path")
    parser.add_argument("--resume", action="store_true",
                       help="resume training from checkpoint")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    print(f"using config file: {args.config}")
    print(f"dataset type: {cfg.dataset_type}")
    print(f"model type: {cfg.model_type}")
    print(f"run mode: {args.mode}")
    
    setup_output_dirs(cfg)
    
    print("creating train dataloader...")
    train_loader = DatasetFactory.create_dataloader(cfg, "train")
    print(f"train dataset size: {len(train_loader.dataset)}")
    
    print("creating val dataloader...")
    val_loader = DatasetFactory.create_dataloader(cfg, "test")
    print(f"val dataset size: {len(val_loader.dataset)}")
    
    print("creating model...")
    model = AudioDrivenTriplaneAvatarLightning(cfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total model parameters: {total_params:,}")
    print(f"trainable model parameters: {trainable_params:,}")
    
    print("creating trainer...")
    trainer = TrainerFactory.create_trainer(cfg)
    
    training_cfg = cfg.training if hasattr(cfg, 'training') else cfg
    print(f"training config:")
    print(f"  max epochs: {training_cfg.max_epochs if hasattr(training_cfg, 'max_epochs') else 'N/A'}")
    print(f"  batch size: {training_cfg.batch_size if hasattr(training_cfg, 'batch_size') else 'N/A'}")
    print(f"  learning rate: {training_cfg.learning_rate if hasattr(training_cfg, 'learning_rate') else 'N/A'}")
    print(f"  total steps: {training_cfg.total_steps if hasattr(training_cfg, 'total_steps') else 'N/A'}")
    
    if args.mode == "train":
        print("start training...")
        
        if cfg.training.resume and cfg.training.ckpt:
            print(f"resume training from checkpoint: {cfg.training.ckpt}")
            checkpoint = torch.load(cfg.training.ckpt)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("loaded model weights")
            
        trainer.fit(model, train_loader, val_loader)
        
        print("training completed, start testing...")
        trainer.test(model, val_loader)
        
    elif args.mode == "test":
        if args.checkpoint is None:
            raise ValueError("test mode requires checkpoint file path")
        
        print(f"loading checkpoint: {args.checkpoint}")

        print("loaded model weights")

        trainer.test(model, val_loader)
        
    elif args.mode == "demo":
        if args.checkpoint is None:
            raise ValueError("demo mode requires checkpoint file path")
        
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            triplane_gaussian_state_dict = {k.replace('triplane_gaussian.', ''): v for k, v in checkpoint['state_dict'].items() 
                                        if k.startswith('triplane_gaussian.') and 'image_feature' not in k}
            model.triplane_gaussian.load_state_dict(triplane_gaussian_state_dict, strict=False)
            print("已加载TriplaneGaussian模型权重（跳过image_feature）")
            
            audio_triplane_state_dict = {k.replace('audio_triplane.', ''): v for k, v in checkpoint['state_dict'].items()
                                        if k.startswith('audio_triplane.')}
            if audio_triplane_state_dict:
                model.audio_triplane.load_state_dict(audio_triplane_state_dict, strict=False)
                print("已加载AudioTriplane模型权重")

        triplane_model = model.triplane_gaussian.to(cfg.device).train()
        audio_model = model.audio_triplane.to(cfg.device).train()

        smplx_model = init_smplx_model(cfg.device)
        renderer = SimpleMeshRenderer(
            smplx_model.faces,
            cfg.dataset.image_size[0],
            cfg.dataset.image_size[1]
        )

        dataset = val_loader.dataset
        collate_function = val_loader.collate_fn

        odd_seq = []
        odd_seq_smplx = []
        even_seq = []
        even_seq_smplx = []

        # 确定要处理的batch索引
        batch_indices = [i for i in range(0, min(320, len(dataset))) if i % 12 == 0]
        print(f"将处理 {len(batch_indices)} 个batch: {batch_indices}")

        # 获取第一个batch用于初始化
        first_data = dataset[0]
        ref_batch, pred_batch, _ = collate_function([first_data])
        
        ref_images = ref_batch.video.to(cfg.device)
        for k, v in ref_batch.smpl_parms.items():
            ref_batch.smpl_parms[k] = v.to(cfg.device)
        for k, v in ref_batch.cam_parms.items():
            ref_batch.cam_parms[k] = v.to(cfg.device)
        
        with torch.no_grad():
            rendered_ref_images, gaussians, triplanes, ref_img_features, pred_smplx, _, smplx_tokens = triplane_model(
                ref_images,
                ref_batch.smpl_parms, 
                ref_batch.cam_parms,
            )

        for batch_idx, idx in enumerate(batch_indices):
            print(f"正在处理第 {batch_idx} 个batch (数据索引 {idx})...")
            
            # 现场获取数据
            data = dataset[idx]
            ref_batch, pred_batch, _ = collate_function([data])
            
            audio_features = pred_batch.audio_features.to(cfg.device)
            pred_images_gt = pred_batch.video.to(cfg.device)
            for k, v in pred_batch.smpl_parms.items():
                pred_batch.smpl_parms[k] = v.to(cfg.device)
            for k, v in pred_batch.cam_parms.items():
                pred_batch.cam_parms[k] = v.to(cfg.device)
                
            with torch.no_grad():
                audio_rendered_images, audio_gaussians, pred_smplx_future, output_triplane_tokens, output_smplx_tokens = audio_model(
                    audio_features,
                    triplanes,
                    ref_img_features,
                    pred_batch.cam_parms,
                    smplx_tokens,
                )

            triplanes = output_triplane_tokens[:, -2:]
            smplx_tokens = output_smplx_tokens[:, -2:]

            # 计算L1 loss
            l1_loss_eval = l1_loss(audio_rendered_images, pred_images_gt.permute(0, 1, 3, 4, 2))
            print(f'Batch {batch_idx} L1 Loss: {l1_loss_eval.item():.6f}')

            # 计算smplx loss
            smplx_loss_eval, losses = smplx_param_loss(pred_smplx_future, pred_batch.smpl_parms)
            print(f'Batch {batch_idx} SMPLX Loss: {smplx_loss_eval.item():.6f}')

            ref_smplx_img = draw_smplx_on_image(
                audio_rendered_images.permute(0, 1, 4, 2, 3),
                pred_smplx_future,
                pred_batch.cam_parms['intrinsic'],
                pred_batch.cam_parms['extrinsic'],
                smplx_model,
                renderer
            )

            # 为每个batch创建单独的文件夹
            batch_output_dir = f'{cfg.training.output_dir}/demo_outputs/batch_{batch_idx:03d}'
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # 获取当前batch的渲染结果
            batch_rendered_images = audio_rendered_images[0].detach().cpu().numpy()
            batch_gt_images = pred_images_gt[0].permute(0, 2, 3, 1).detach().cpu().numpy()
            

            # 保存当前batch的每一帧
            T = batch_rendered_images.shape[0]
            for t in range(T):
                plt.imsave(f'{batch_output_dir}/frame_{t:03d}.png', batch_rendered_images[t])
                plt.imsave(f'{batch_output_dir}/gt_frame_{t:03d}.png', batch_gt_images[t])
                plt.imsave(f'{batch_output_dir}/ref_smplx_frame_{t:03d}.png', ref_smplx_img[t])
                even_seq.append(batch_rendered_images[t])
                even_seq_smplx.append(ref_smplx_img[t])
            print(f'Batch {batch_idx} 的渲染结果已保存到 {batch_output_dir}')

        #######################33
        batch_indices = [i for i in range(0, min(320, len(dataset))) if i % 12 == 1]
        print(f"将处理 {len(batch_indices)} 个batch: {batch_indices}")

        # 获取第一个batch用于初始化
        first_data = dataset[1]
        ref_batch, pred_batch, _ = collate_function([first_data])
        
        ref_images = ref_batch.video.to(cfg.device)
        for k, v in ref_batch.smpl_parms.items():
            ref_batch.smpl_parms[k] = v.to(cfg.device)
        for k, v in ref_batch.cam_parms.items():
            ref_batch.cam_parms[k] = v.to(cfg.device)
        
        with torch.no_grad():
            rendered_ref_images, gaussians, triplanes, ref_img_features, pred_smplx, _, smplx_tokens = triplane_model(
                ref_images,
                ref_batch.smpl_parms, 
                ref_batch.cam_parms,
            )

        for batch_idx, idx in enumerate(batch_indices):
            print(f"正在处理第 {batch_idx} 个batch (数据索引 {idx})...")
            
            # 现场获取数据
            data = dataset[idx]
            ref_batch, pred_batch, _ = collate_function([data])
            
            audio_features = pred_batch.audio_features.to(cfg.device)
            pred_images_gt = pred_batch.video.to(cfg.device)
            for k, v in pred_batch.smpl_parms.items():
                pred_batch.smpl_parms[k] = v.to(cfg.device)
            for k, v in pred_batch.cam_parms.items():
                pred_batch.cam_parms[k] = v.to(cfg.device)
                
            with torch.no_grad():
                audio_rendered_images, audio_gaussians, pred_smplx_future, output_triplane_tokens, output_smplx_tokens = audio_model(
                    audio_features,
                    triplanes,
                    ref_img_features,
                    pred_batch.cam_parms,
                    smplx_tokens,
                )

            triplanes = output_triplane_tokens[:, -2:]
            smplx_tokens = output_smplx_tokens[:, -2:]

            # 计算L1 loss
            l1_loss_eval = l1_loss(audio_rendered_images, pred_images_gt.permute(0, 1, 3, 4, 2))
            print(f'Batch {batch_idx} L1 Loss: {l1_loss_eval.item():.6f}')

            # 计算smplx loss
            smplx_loss_eval, losses = smplx_param_loss(pred_smplx_future, pred_batch.smpl_parms)
            print(f'Batch {batch_idx} SMPLX Loss: {smplx_loss_eval.item():.6f}')

            ref_smplx_img = draw_smplx_on_image(
                audio_rendered_images.permute(0, 1, 4, 2, 3),
                pred_smplx_future,
                pred_batch.cam_parms['intrinsic'],
                pred_batch.cam_parms['extrinsic'],
                smplx_model,
                renderer
            )

            # 为每个batch创建单独的文件夹
            batch_output_dir = f'{cfg.training.output_dir}/demo_outputs/batch_{batch_idx:03d}'
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # 获取当前batch的渲染结果
            batch_rendered_images = audio_rendered_images[0].detach().cpu().numpy()
            batch_gt_images = pred_images_gt[0].permute(0, 2, 3, 1).detach().cpu().numpy()
            
            # 保存当前batch的每一帧
            T = batch_rendered_images.shape[0]
            for t in range(T):
                odd_seq.append(batch_rendered_images[t])
                odd_seq_smplx.append(ref_smplx_img[t])
            
            print(f'Batch {batch_idx} 的渲染结果已保存到 {batch_output_dir}')
        
        
        # 交替组合序列
        combined_seq = []
        combined_seq_smplx = []
        for i in range(len(even_seq)):
            combined_seq.append(even_seq[i])
            combined_seq_smplx.append(even_seq_smplx[i])
            if i < len(odd_seq):
                combined_seq.append(odd_seq[i])
                combined_seq_smplx.append(odd_seq_smplx[i])
        
        import cv2
        import numpy as np
        
        # 获取视频尺寸
        height, width = combined_seq[0].shape[:2]
        height_smplx, width_smplx = combined_seq_smplx[0].shape[:2]

        audio_path = '/home/liubingqi/work/data/clip_2m52s_to_3m10s/audio_2m52s_to_3m10s.wav'
        
        # 创建两个视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_original = cv2.VideoWriter(f'{cfg.training.output_dir}/demo_outputs/original_sequence.mp4', 
                            fourcc, 24.0, (width, height))
        out_smplx = cv2.VideoWriter(f'{cfg.training.output_dir}/demo_outputs/smplx_sequence.mp4',
                            fourcc, 24.0, (width_smplx, height_smplx))
        
        # 写入每一帧
        for frame, frame_smplx in zip(combined_seq, combined_seq_smplx):
            # 处理原始帧
            frame_uint8 = (frame * 255).astype(np.uint8)
            if frame_uint8.shape[-1] == 3:
                frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            out_original.write(frame_uint8)
            
            # 处理SMPLX帧
            frame_smplx_uint8 = (frame_smplx * 255).astype(np.uint8)
            if frame_smplx_uint8.shape[-1] == 3:
                frame_smplx_uint8 = cv2.cvtColor(frame_smplx_uint8, cv2.COLOR_RGB2BGR)
            out_smplx.write(frame_smplx_uint8)
        
        # 释放视频写入器
        out_original.release()
        out_smplx.release()
        
        # 使用ffmpeg添加音频
        import subprocess
        
        # 计算视频时长
        video_duration = len(combined_seq) / 24.0
        
        # 为两个视频添加音频
        for video_name in ['original_sequence', 'smplx_sequence']:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', f'{cfg.training.output_dir}/demo_outputs/{video_name}.mp4',
                '-i', audio_path,
                '-t', str(video_duration),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                f'{cfg.training.output_dir}/demo_outputs/{video_name}_with_audio.mp4'
            ]
            subprocess.run(ffmpeg_cmd)
        
        print(f'所有渲染结果已保存到 demo_outputs 文件夹，并已添加音频')
            
        print("program executed successfully!")

if __name__ == "__main__":
    main()
