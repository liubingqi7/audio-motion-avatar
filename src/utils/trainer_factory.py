import os
import torch
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

class TrainerFactory:
    """训练器工厂类，用于根据配置创建训练器"""
    
    @staticmethod
    def create_trainer(cfg: DictConfig):
        """
        根据配置创建训练器
        
        Args:
            cfg: 配置对象
            
        Returns:
            Lightning训练器对象
        """
        logger = TrainerFactory._create_logger(cfg)
        
        callbacks = TrainerFactory._create_callbacks(cfg)
        
        # 从嵌套配置中提取训练参数
        training_cfg = cfg.training if hasattr(cfg, 'training') else cfg
        
        trainer = L.Trainer(
            max_epochs=training_cfg.max_epochs if hasattr(training_cfg, 'max_epochs') else 100,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=training_cfg.devices if hasattr(training_cfg, 'devices') else 1,
            strategy=training_cfg.strategy if hasattr(training_cfg, 'strategy') else "auto",
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=training_cfg.log_every_n_steps if hasattr(training_cfg, 'log_every_n_steps') else 10,
            check_val_every_n_epoch=training_cfg.val_every_n_epoch if hasattr(training_cfg, 'val_every_n_epoch') else 1,
            default_root_dir=training_cfg.output_dir if hasattr(training_cfg, 'output_dir') else "./outputs",
            deterministic=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            fast_dev_run=training_cfg.fast_dev_run if hasattr(training_cfg, 'fast_dev_run') else False,
            accumulate_grad_batches=training_cfg.gradient_accumulate_steps if hasattr(training_cfg, 'gradient_accumulate_steps') else 1,
        )
        
        return trainer
    
    @staticmethod
    def _create_logger(cfg: DictConfig):
        # 从嵌套配置中提取日志参数
        training_cfg = cfg.training if hasattr(cfg, 'training') else cfg
        
        use_wandb = training_cfg.use_wandb if hasattr(training_cfg, 'use_wandb') else False
        
        if use_wandb:
            return WandbLogger(
                project=training_cfg.wandb_project if hasattr(training_cfg, 'wandb_project') else "audio_motion_avatar",
                name=training_cfg.wandb_run_name if hasattr(training_cfg, 'wandb_run_name') else None,
                log_model=True
            )
        else:
            output_dir = training_cfg.output_dir if hasattr(training_cfg, 'output_dir') else "./outputs"
            experiment_name = cfg.experiment_name if hasattr(cfg, 'experiment_name') else "experiment"
            log_dir = os.path.join(output_dir, "tensorboard_logs")
            return TensorBoardLogger(
                save_dir=log_dir,
                name=experiment_name
            )
    
    @staticmethod
    def _create_callbacks(cfg: DictConfig):
        callbacks = []
        
        # 从嵌套配置中提取参数
        training_cfg = cfg.training if hasattr(cfg, 'training') else cfg
        output_dir = training_cfg.output_dir if hasattr(training_cfg, 'output_dir') else "./outputs"
        experiment_name = cfg.experiment_name if hasattr(cfg, 'experiment_name') else "experiment"
        
        # 从配置中获取monitor设置，默认为"val/loss"
        monitor = "val/loss_total"
        mode = "min"
        save_top_k = 2
        
        if hasattr(training_cfg, 'validation'):
            validation_cfg = training_cfg.validation
            if hasattr(validation_cfg, 'monitor'):
                monitor = validation_cfg.monitor
            if hasattr(validation_cfg, 'mode'):
                mode = validation_cfg.mode
            if hasattr(validation_cfg, 'save_top_k'):
                save_top_k = validation_cfg.save_top_k
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename=f"{experiment_name}-{{epoch:02d}}-{{val_loss_total:.4f}}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=True,
            every_n_epochs=1
        )
        callbacks.append(checkpoint_callback)
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        return callbacks