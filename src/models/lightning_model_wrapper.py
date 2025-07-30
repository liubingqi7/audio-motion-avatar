import torch
import torch.nn as nn
import lightning as L
import wandb
import os
import torchvision.utils as vutils
import time
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import random

from omegaconf import DictConfig
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, chamfer_distance

from src.utils.loss_utils import l1_loss, ssim, smplx_param_loss, LPIPS

from src.models.triplane_net import SMPLXTriplaneEncoder, FeatureFusionNetwork
from src.models.triplane_audio_net import AudioTriplaneNet
from src.models.image_feature import ImageFeature, SapiensWrapper
from src.models.smplx_decoder import SMPLXDecoder
from src.models.renderer import Renderer, render_multi_view
from src.configs.config_loader import ConfigLoader

class TriplaneGaussianAvatar(nn.Module):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.cfg = cfg
        
        model_cfg = ConfigLoader.create_model_config(cfg)

        self.sapiens_encoder = SapiensWrapper(model_cfg)
        self.image_feature = ImageFeature()
        self.smplx_decoder = SMPLXDecoder(model_cfg)

        self.smplx_triplane_encoder = SMPLXTriplaneEncoder(model_cfg, self.smplx_decoder)
        self.fusion_network = FeatureFusionNetwork(model_cfg)

        self.renderer = Renderer(model_cfg, self.smplx_decoder)

    def forward(self, img, smpl_params_gt, cam_params):
        B, T = img.shape[:2]
        image_tokens = self.sapiens_encoder(img.reshape(B*T, *img.shape[2:]))
        image_tokens = image_tokens.reshape(B, T, -1, image_tokens.shape[-1])
        image_features = self.image_feature(img, image_tokens)

        smplx_triplane, smpl_tokens, pred_smpl_1 = self.smplx_triplane_encoder(cam_params, image_tokens, smpl_params_gt, image_features)

        fused_triplane_tokens, smpl_tokens = self.fusion_network(smplx_triplane, image_tokens, smpl_tokens)
        
        rendered_images, gaussians, pred_smpl_2 = self.renderer(fused_triplane_tokens, cam_params, smpl_tokens, smpl_params_gt)

        return rendered_images, gaussians, fused_triplane_tokens, image_tokens, pred_smpl_1, pred_smpl_2, smpl_tokens
    

class TriplaneGaussianAvatarLightning(L.LightningModule):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.cfg = cfg
        
        training_cfg = cfg.training if hasattr(cfg, 'training') else cfg
        self.lr = training_cfg.learning_rate if hasattr(training_cfg, 'learning_rate') else 1e-4
        self.total_steps = training_cfg.total_steps if hasattr(training_cfg, 'total_steps') else 200000
        
        self.triplane_gaussian = TriplaneGaussianAvatar(cfg)
        self.save_hyperparameters(cfg)
        self.writer = None  # TensorBoard writer
        self.use_wandb = training_cfg.use_wandb if hasattr(training_cfg, 'use_wandb') else False

        # self.lpips_loss = LPIPS()
        self.l1_loss = l1_loss
        self.ssim_loss = ssim
        self.smplx_param_loss = smplx_param_loss

    def on_save_checkpoint(self, checkpoint) -> None:
        state_dict = checkpoint.get("state_dict", {})
        keys_to_remove = [k for k in state_dict if k.startswith("triplane_gaussian.sapiens_encoder.")]
        for k in keys_to_remove:
            state_dict.pop(k, None)
        checkpoint["state_dict"] = state_dict

    def training_step(self, batch, batch_idx):
        ref_batch, test_batch, _ = batch
        
        
        ref_images = ref_batch.video.to(self.device)
        test_images = test_batch.video.to(self.device)

        for k, v in ref_batch.smpl_parms.items():
            ref_batch.smpl_parms[k] = v.to(self.device)
        for k, v in ref_batch.cam_parms.items():
            ref_batch.cam_parms[k] = v.to(self.device)

        for k, v in test_batch.smpl_parms.items():
            test_batch.smpl_parms[k] = v.to(self.device)
        for k, v in test_batch.cam_parms.items():
            test_batch.cam_parms[k] = v.to(self.device)

        (
            rendered_images, 
            gaussians, 
            fused_triplane_tokens, 
            image_tokens, 
            pred_smpl_1, 
            pred_smpl_2, 
            smpl_tokens
        ) = self.triplane_gaussian(ref_images, ref_batch.smpl_parms, ref_batch.cam_parms)

        rendered_images_target = None
        if test_images is not None:
            B, T = test_images.shape[0], test_images.shape[1]
            
            intrinsic_list = []
            extrinsic_list = []
            
            for b in range(B):
                for t in range(T):
                    curr_intrinsic = test_batch.cam_parms['intrinsic'][b:b+1, t:t+1]
                    curr_extrinsic = test_batch.cam_parms['extrinsic'][b:b+1, t:t+1]
                    
                    intrinsic_list.append(curr_intrinsic)
                    extrinsic_list.append(curr_extrinsic)

            intrinsic = torch.cat(intrinsic_list, dim=1).reshape(B, T, 3, 3)
            extrinsic = torch.cat(extrinsic_list, dim=1).reshape(B, T, 4, 4)

            args = type('Args', (), {
                'image_size': self.cfg.model.renderer.image_size,
                'rgb': True,
                'sh_degree': 3
            })()
            rendered_images_target = render_multi_view(gaussians, intrinsic, extrinsic, args)

        losses = {}
        
        loss_smplx = smplx_param_loss(pred_smpl_1, ref_batch.smpl_parms)[0] + smplx_param_loss(pred_smpl_2, ref_batch.smpl_parms)[0]
        
        losses['l1_train'] = l1_loss(rendered_images, ref_images.permute(0, 1, 3, 4, 2))
        losses['ssim_train'] = 1 - ssim(rendered_images, ref_images.permute(0, 1, 3, 4, 2))
        loss_train = losses['l1_train'] + 0.1 * losses['ssim_train']
        
        if test_images is not None and rendered_images_target is not None:
            losses['l1_test'] = l1_loss(rendered_images_target, test_images.permute(0, 1, 3, 4, 2))
            losses['ssim_test'] = 1 - ssim(rendered_images_target, test_images.permute(0, 1, 3, 4, 2))
            loss_test = losses['l1_test'] + 0.1 * losses['ssim_test']
        else:
            losses['l1_test'] = torch.tensor(0.0, device=self.device)
            losses['ssim_test'] = torch.tensor(0.0, device=self.device)
            loss_test = torch.tensor(0.0, device=self.device)

        pred_xyz = gaussians['xyz']
        pcd_points = ref_batch.pcd_points.to(self.device)
        chamfer_loss, _ = chamfer_distance(pred_xyz, pcd_points)
        self.log('train/chamfer_loss', chamfer_loss, prog_bar=True)

        loss = loss_train + loss_test + 0.01 * loss_smplx + 50 * chamfer_loss
        
        self.log('train/l1_loss_train', losses['l1_train'], prog_bar=True)
        self.log('train/ssim_loss_train', losses['ssim_train'], prog_bar=False)
        self.log('train/l1_loss_test', losses['l1_test'], prog_bar=True)
        self.log('train/ssim_loss_test', losses['ssim_test'], prog_bar=False)
        self.log('train/smplx_param_loss', loss_smplx, prog_bar=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('train/learning_rate', current_lr, prog_bar=False)
        self.log('train/loss', loss, prog_bar=False)

        if self.global_step % 500 == 0 and self.global_rank == 0:
            self._save_training_images(ref_images, rendered_images, batch_idx, test_images, rendered_images_target)

        return loss

    def _save_training_images(self, ref_images, rendered_images, batch_idx, test_images=None, rendered_images_target=None):
        B, T, _, H, W = ref_images.shape
        
        all_images = []
        ref_images_permuted = ref_images.permute(0, 1, 3, 4, 2).reshape(1, B, T, H, W, 3)
        rendered_images_reshaped = rendered_images.reshape(1, B, T, H, W, 3)
        
        for frame_idx in range(ref_images_permuted.shape[1]):
            for t in range(T):
                real_image = ref_images_permuted[0, frame_idx, t]
                rendered_image = rendered_images_reshaped[0, frame_idx, t]
                combined = torch.cat([rendered_image, real_image], dim=1)
            all_images.append(combined)
        
        images = torch.cat(all_images, dim=0)
        
        output_dir = self.cfg.training.output_dir if hasattr(self.cfg, 'training') and hasattr(self.cfg.training, 'output_dir') else "./outputs"
        os.makedirs(os.path.join(output_dir, "train_images"), exist_ok=True)
        
        images_chw = images.detach().cpu().permute(2, 0, 1)
        vutils.save_image(
            images_chw,
            os.path.join(output_dir, f"train_images/triplane_comparison_{self.global_step}.png"),
            normalize=True
        )
        
        if test_images is not None and rendered_images_target is not None:
            target_all_images = []
            for frame_idx in range(test_images.shape[1]):
                real_target_image = test_images[0, frame_idx].permute(1, 2, 0)
                rendered_target_image = rendered_images_target[0, frame_idx]
                target_combined = torch.cat([rendered_target_image, real_target_image], dim=1)
                target_all_images.append(target_combined)
            
            target_images_combined = torch.cat(target_all_images, dim=0)
            
            os.makedirs(os.path.join(output_dir, "target_images"), exist_ok=True)
            target_images_chw = target_images_combined.detach().cpu().permute(2, 0, 1)
            vutils.save_image(
                target_images_chw,
                os.path.join(output_dir, f"target_images/target_comparison_{self.global_step}.png"),
                normalize=True
            )

    def test_step(self, batch, batch_idx):
        self.triplane_gaussian.train() 
        ref_batch, test_batch, _ = batch
        
        ref_images = ref_batch.video.to(self.device)
        test_images = test_batch.video.to(self.device)

        for k, v in ref_batch.smpl_parms.items():
            ref_batch.smpl_parms[k] = v.to(self.device)
        for k, v in ref_batch.cam_parms.items():
            ref_batch.cam_parms[k] = v.to(self.device)

        for k, v in test_batch.smpl_parms.items():
            test_batch.smpl_parms[k] = v.to(self.device)
        for k, v in test_batch.cam_parms.items():
            test_batch.cam_parms[k] = v.to(self.device)

        (
            rendered_images, 
            gaussians, 
            fused_triplane_tokens, 
            image_tokens, 
            pred_smpl_1, 
            pred_smpl_2, 
            smpl_tokens
        ) = self.triplane_gaussian(ref_images, ref_batch.smpl_parms, ref_batch.cam_parms)

        rendered_images_target = None
        if test_images is not None:
            B, T = test_images.shape[0], test_images.shape[1]
            
            intrinsic_list = []
            extrinsic_list = []
            
            for b in range(B):
                for t in range(T):
                    curr_intrinsic = test_batch.cam_parms['intrinsic'][b:b+1, t:t+1]
                    curr_extrinsic = test_batch.cam_parms['extrinsic'][b:b+1, t:t+1]
                    
                    intrinsic_list.append(curr_intrinsic)
                    extrinsic_list.append(curr_extrinsic)

            intrinsic = torch.cat(intrinsic_list, dim=1).reshape(B, T, 3, 3)
            extrinsic = torch.cat(extrinsic_list, dim=1).reshape(B, T, 4, 4)

            args = type('Args', (), {
                'image_size': self.cfg.model.renderer.image_size,
                'rgb': True,
                'sh_degree': 3
            })()
            rendered_images_target = render_multi_view(gaussians, intrinsic, extrinsic, args)
        
        losses = {}
        
        losses['smplx_param_loss'] = smplx_param_loss(pred_smpl_1, ref_batch.smpl_parms)[0] + smplx_param_loss(pred_smpl_2, ref_batch.smpl_parms)[0]
        
        losses['l1_train'] = l1_loss(rendered_images, ref_images.permute(0, 1, 3, 4, 2))
        losses['ssim_train'] = 1 - ssim(rendered_images, ref_images.permute(0, 1, 3, 4, 2))
        
        if test_images is not None and rendered_images_target is not None:
            losses['l1_test'] = l1_loss(rendered_images_target, test_images.permute(0, 1, 3, 4, 2))
            losses['ssim_test'] = 1 - ssim(rendered_images_target, test_images.permute(0, 1, 3, 4, 2))
        else:
            losses['l1_test'] = torch.tensor(0.0, device=self.device)
            losses['ssim_test'] = torch.tensor(0.0, device=self.device)

        self._save_test_images(ref_images, rendered_images, batch_idx, test_images, rendered_images_target)

        # print(f"test/l1_loss_train: {losses['l1_train']}, test/ssim_loss_train: {losses['ssim_train']}, test/l1_loss_test: {losses['l1_test']}, test/ssim_loss_test: {losses['ssim_test']}, test/smplx_param_loss: {loss_smplx}")

        self.log_dict({f"test/{k}": v for k, v in losses.items()}, prog_bar=True, on_step=False, on_epoch=True)

        return losses

    def _save_test_images(self, ref_images, rendered_images, batch_idx, test_images=None, rendered_images_target=None):
        B, T, _, H, W = ref_images.shape
        
        all_images = []
        ref_images_permuted = ref_images.permute(0, 1, 3, 4, 2).reshape(1, B, T, H, W, 3)
        rendered_images_reshaped = rendered_images.reshape(1, B, T, H, W, 3)
        
        for frame_idx in range(ref_images_permuted.shape[1]):
            for t in range(T):
                real_image = ref_images_permuted[0, frame_idx, t]
                rendered_image = rendered_images_reshaped[0, frame_idx, t]
                combined = torch.cat([rendered_image, real_image], dim=1)
            all_images.append(combined)
        
        images = torch.cat(all_images, dim=0)
        
        output_dir = self.cfg.training.output_dir if hasattr(self.cfg, 'training') and hasattr(self.cfg.training, 'output_dir') else "./outputs"
        os.makedirs(os.path.join(output_dir, "test_images"), exist_ok=True)
        
        images_chw = images.detach().cpu().permute(2, 0, 1)
        vutils.save_image(
            images_chw,
            os.path.join(output_dir, f"test_images/triplane_comparison_{batch_idx}.png"),
            normalize=True
        )
        
        if test_images is not None and rendered_images_target is not None:
            target_all_images = []
            for frame_idx in range(test_images.shape[1]):
                real_target_image = test_images[0, frame_idx].permute(1, 2, 0)
                rendered_target_image = rendered_images_target[0, frame_idx]
                target_combined = torch.cat([rendered_target_image, real_target_image], dim=1)
                target_all_images.append(target_combined)
            
            target_images_combined = torch.cat(target_all_images, dim=0)
            
            os.makedirs(os.path.join(output_dir, "target_images"), exist_ok=True)
            target_images_chw = target_images_combined.detach().cpu().permute(2, 0, 1)
            vutils.save_image(
                target_images_chw,
                os.path.join(output_dir, f"test_images/target_comparison_{batch_idx}.png"),
                normalize=True
            )
        

    def validation_step(self, batch, batch_idx):
        ref_batch, test_batch, _ = batch
        
        for k, v in ref_batch.smpl_parms.items():
            ref_batch.smpl_parms[k] = v.to(self.device)
        for k, v in ref_batch.cam_parms.items():
            ref_batch.cam_parms[k] = v.to(self.device)

        ref_images = ref_batch.video.to(self.device)

        outputs = self.triplane_gaussian(ref_images, ref_batch.smpl_parms, ref_batch.cam_parms)
        rendered_images, gaussians, fused_triplane_tokens, image_tokens, pred_smpl_1, pred_smpl_2, smpl_tokens = outputs

        loss_dict = {}
        loss_dict["l1"] = self.l1_loss(rendered_images, ref_images.permute(0, 1, 3, 4, 2))
        loss_dict["ssim"] = self.ssim_loss(rendered_images, ref_images.permute(0, 1, 3, 4, 2))
        # loss_dict["lpips"] = self.lpips_loss(rendered_images, ref_images.permute(0, 1, 3, 4, 2))
        loss_dict["total"] = loss_dict["l1"] + 0.1 * loss_dict["ssim"] # + 0.1 * loss_dict["lpips"]

        # 添加val/loss用于ModelCheckpoint监控
        loss_dict["loss"] = loss_dict["total"]

        self.log("val/loss_l1_train", loss_dict["l1"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/loss_ssim_train", loss_dict["ssim"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/loss_total", loss_dict["total"], prog_bar=True, on_step=False, on_epoch=True)

        # print(f"val/loss_l1_train: {loss_dict['l1']}, val/loss_ssim_train: {loss_dict['ssim']}, val/loss_total: {loss_dict['total']}")

        return loss_dict["total"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=self.total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def on_train_start(self):
        if not self.use_wandb and self.logger is not None and hasattr(self.logger, "log_dir"):
            self.writer = SummaryWriter(self.logger.log_dir)

    def on_train_end(self):
        if self.writer is not None:
            self.writer.close()

class AudioDrivenTriplaneAvatarLightning(L.LightningModule):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.cfg = cfg
        
        training_cfg = cfg.training if hasattr(cfg, 'training') else cfg
        self.lr = training_cfg.learning_rate if hasattr(training_cfg, 'learning_rate') else 1e-4
        self.total_steps = training_cfg.total_steps if hasattr(training_cfg, 'total_steps') else 200000
        
        self.save_hyperparameters(cfg)
        self.writer = None  # TensorBoard writer
        self.use_wandb = training_cfg.use_wandb if hasattr(training_cfg, 'use_wandb') else False
                
        self.triplane_gaussian = TriplaneGaussianAvatar(cfg)
        self.audio_triplane = AudioTriplaneNet(cfg, self.triplane_gaussian.renderer)

        self.prediction_cache = {}  # {(subject_id, frame_id): {'triplane': tensor, 'smplx_tokens': tensor}}
        self.cache_replacement_prob = getattr(cfg, 'cache_replacement_prob', 0.0)

    def on_save_checkpoint(self, checkpoint) -> None:
        state_dict = checkpoint.get("state_dict", {})
        keys_to_remove = [k for k in state_dict if k.startswith("triplane_gaussian.sapiens_encoder.")]
        for k in keys_to_remove:
            state_dict.pop(k, None)
        checkpoint["state_dict"] = state_dict

    def training_step(self, batch, batch_idx):
        ref_batch, pred_batch, batch_id = batch
        
        for k, v in ref_batch.smpl_parms.items():
            ref_batch.smpl_parms[k] = v.to(self.device)
        for k, v in ref_batch.cam_parms.items():
            ref_batch.cam_parms[k] = v.to(self.device)
        
        for k, v in pred_batch.smpl_parms.items():
            pred_batch.smpl_parms[k] = v.to(self.device)
        for k, v in pred_batch.cam_parms.items():
            pred_batch.cam_parms[k] = v.to(self.device)
            
        ref_images = ref_batch.video.to(self.device)
        audio_features = pred_batch.audio_features.to(self.device)
        
        # reconstruct triplane and smplx_tokens from input images
        with torch.no_grad():
            rendered_images, gaussians, triplanes, ref_img_features, pred_smplx, _,  smplx_tokens = self.triplane_gaussian(
                ref_images,
                ref_batch.smpl_parms, 
                ref_batch.cam_parms,
            )

        # replace generated triplane and smplx_tokens with cached ones
        use_cache = 0
        if self.cache_replacement_prob > 0 and random.random() < self.cache_replacement_prob:
            subject_id = 0
            cache_frame_id = batch_id
            
            if (subject_id, cache_frame_id) in self.prediction_cache:
                cached_data = self.prediction_cache[(subject_id, cache_frame_id)]
                
                if (cached_data['triplane'] is not None and 
                    cached_data['smplx_tokens'] is not None):
                    triplanes = cached_data['triplane'].to(self.device)
                    smplx_tokens = cached_data['smplx_tokens'].to(self.device)
                    use_cache = cached_data['iter']

                    # print(f"Using cached triplane and smplx_tokens at step {self.global_step}, current batch_id: {batch_id}, loading cache from {cache_frame_id}, iter: {use_cache}")

        # predict future smplx_tokens and triplane_tokens from audio features
        audio_rendered_images, audio_gaussians, pred_smplx_future, output_triplane_tokens, output_smplx_tokens = self.audio_triplane(
            audio_features,
            triplanes,
            ref_img_features,
            pred_batch.cam_parms,
            smplx_tokens,
        )

        new_cache_item = None
        if self.cache_replacement_prob > 0 and use_cache < 30:
            subject_id = 0
            future_frame_id = batch_id + 12
            key = (subject_id, future_frame_id)
            cache_data = {
                'triplane': output_triplane_tokens[:, -2:].clone().detach().cpu(),
                'smplx_tokens': output_smplx_tokens[:, -2:].clone().detach().cpu(),
                'iter': use_cache + 1
            }
            self.prediction_cache[key] = cache_data
            new_cache_item = {key: cache_data}

        if dist.is_available() and dist.is_initialized():
            if new_cache_item is None:
                new_cache_item = {}

            world_size = dist.get_world_size()
            gathered = [None] * world_size

            dist.all_gather_object(gathered, new_cache_item)

            for rank_dict in gathered:
                if not rank_dict:
                    continue
                self.prediction_cache.update(rank_dict)

        losses = {}
        losses['l1_target'] = l1_loss(audio_rendered_images, pred_batch.video.permute(0, 1, 3, 4, 2))
        
        suffix_dict = {0: "no_cache", 1: "cache_1", 2: "cache_2"}
        
        for k, s in suffix_dict.items():
            name = f"train/l1_loss_{s}"
            val = losses['l1_target'] if k == use_cache else float("nan")
            self.log(
                name,
                val,
                on_step=True,  
                on_epoch=False
            )

        losses['ssim_target'] = 1 - ssim(audio_rendered_images, pred_batch.video.permute(0, 1, 3, 4, 2))
        loss_target = losses['l1_target'] + 0.1 * losses['ssim_target']

        # losses['l1_ref'] = l1_loss(rendered_images, ref_images.permute(0, 1, 3, 4, 2))
        # losses['ssim_ref'] = 1 - ssim(rendered_images, ref_images.permute(0, 1, 3, 4, 2))
        # loss_ref = losses['l1_ref'] + 0.1 * losses['ssim_ref']

        smpl_loss_ref = 0
        smpl_loss_future = 0
        
        smpl_loss_future = smplx_param_loss(pred_smplx_future, pred_batch.smpl_parms)[0]

        total_loss = 10 * loss_target + 0.05 * smpl_loss_future

        self.log('train/l1_loss_target', losses['l1_target'], prog_bar=True)
        self.log('train/ssim_loss_target', losses['ssim_target'], prog_bar=False)
        self.log('train/loss_target', loss_target, prog_bar=False)
        
        # self.log('train/l1_loss_ref', losses['l1_ref'], prog_bar=True)
        # self.log('train/ssim_loss_ref', losses['ssim_ref'], prog_bar=False)
        # self.log('train/loss_ref', loss_ref, prog_bar=False)
        
        self.log('train/smpl_loss_ref', smpl_loss_ref, prog_bar=True)
        self.log('train/smpl_loss_future', smpl_loss_future, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=False)

        if self.global_step % 200 == 0 and self.global_rank == 0:
            target_all_images = []                
            for frame_idx in range(pred_batch.video.shape[1]):
                real_target_image = pred_batch.video[0, frame_idx].permute(1, 2, 0)
                rendered_target_image = audio_rendered_images[0, frame_idx] 
                target_combined = torch.cat([rendered_target_image, real_target_image], dim=1)
                target_all_images.append(target_combined)
            
            target_images_combined = torch.cat(target_all_images, dim=0)
            
            ref_all_images = []
            for frame_idx in range(ref_images.shape[1]):
                real_ref_image = ref_images[0, frame_idx].permute(1, 2, 0)
                rendered_ref_image = rendered_images[0, frame_idx]
                ref_combined = torch.cat([rendered_ref_image, real_ref_image], dim=1)
                ref_all_images.append(ref_combined)
                
            ref_images_combined = torch.cat(ref_all_images, dim=0)
        
            output_dir = self.cfg.training.output_dir if hasattr(self.cfg, 'training') and hasattr(self.cfg.training, 'output_dir') else "./outputs"
            os.makedirs(os.path.join(output_dir, "audio_rendered_images"), exist_ok=True)
            
            target_images_chw = target_images_combined.detach().cpu().permute(2, 0, 1)
            vutils.save_image(
                target_images_chw,
                os.path.join(output_dir, f"audio_rendered_images/audio_comparison_{self.global_step}_iter{use_cache}.png"),
                normalize=True
            )
            
            ref_images_chw = ref_images_combined.detach().cpu().permute(2, 0, 1)
            vutils.save_image(
                ref_images_chw,
                os.path.join(output_dir, f"audio_rendered_images/ref_comparison_{self.global_step}.png"),
                normalize=True
            )

        return total_loss
    
    def predict_step(self, batch, batch_idx):
        ref_batch, pred_batch, batch_id = batch
        
        for k, v in ref_batch.smpl_parms.items():
            ref_batch.smpl_parms[k] = v.to(self.device)
        for k, v in ref_batch.cam_parms.items():
            ref_batch.cam_parms[k] = v.to(self.device)
        
        for k, v in pred_batch.smpl_parms.items():
            pred_batch.smpl_parms[k] = v.to(self.device)
        for k, v in pred_batch.cam_parms.items():
            pred_batch.cam_parms[k] = v.to(self.device)
            
        ref_images = ref_batch.video.to(self.device)
        audio_features = pred_batch.audio_features.to(self.device)
        
        # reconstruct triplane and smplx_tokens from input images
        with torch.no_grad():
            rendered_images, gaussians, triplanes, ref_img_features, pred_smplx, _,  smplx_tokens = self.triplane_gaussian(
                ref_images,
                ref_batch.smpl_parms, 
                ref_batch.cam_parms,
            )

        # predict future smplx_tokens and triplane_tokens from audio features
        audio_rendered_images, audio_gaussians, pred_smplx_future, output_triplane_tokens, output_smplx_tokens = self.audio_triplane(
            audio_features,
            triplanes,
            ref_img_features,
            pred_batch.cam_parms,
            smplx_tokens,
        )

        output_dir = self.cfg.training.output_dir if hasattr(self.cfg, 'training') and hasattr(self.cfg.training, 'output_dir') else "./outputs"
        os.makedirs(os.path.join(output_dir, "predict_results"), exist_ok=True)
        
        target_all_images = []                
        for frame_idx in range(pred_batch.video.shape[1]):
            real_target_image = pred_batch.video[0, frame_idx].permute(1, 2, 0)
            rendered_target_image = audio_rendered_images[0, frame_idx] 
            target_combined = torch.cat([rendered_target_image, real_target_image], dim=1)
            target_all_images.append(target_combined)
        
        target_images_combined = torch.cat(target_all_images, dim=0)
        target_images_chw = target_images_combined.detach().cpu().permute(2, 0, 1)
        vutils.save_image(
            target_images_chw,
            os.path.join(output_dir, f"predict_results/audio_comparison_{batch_idx}.png"),
            normalize=True
        )
        
        ref_all_images = []
        for frame_idx in range(ref_images.shape[1]):
            real_ref_image = ref_images[0, frame_idx].permute(1, 2, 0)
            rendered_ref_image = rendered_images[0, frame_idx]
            ref_combined = torch.cat([rendered_ref_image, real_ref_image], dim=1)
            ref_all_images.append(ref_combined)
            
        ref_images_combined = torch.cat(ref_all_images, dim=0)
        ref_images_chw = ref_images_combined.detach().cpu().permute(2, 0, 1)
        vutils.save_image(
            ref_images_chw,
            os.path.join(output_dir, f"predict_results/ref_comparison_{batch_idx}.png"),
            normalize=True
        )
        
        return audio_rendered_images
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=self.total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }