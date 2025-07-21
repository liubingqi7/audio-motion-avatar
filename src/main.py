import os
import torch
import lightning as L
from omegaconf import OmegaConf, DictConfig
import argparse

from src.datasets.dataset_factory import DatasetFactory
from src.models.lightning_model_wrapper import TriplaneGaussianAvatarLightning
from src.utils.trainer_factory import TrainerFactory
from src.configs.config_loader import ConfigLoader

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
    output_dir = training_cfg.output_dir if hasattr(training_cfg, 'output_dir') else "./outputs"
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
    parser.add_argument("--config", type=str, default="src/configs/config_thuman.yaml", 
                       help="config file path")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "test", "predict"],
                       help="run mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="checkpoint file path")
    parser.add_argument("--resume", action="store_true",
                       help="resume training from checkpoint")
    
    parser.add_argument("overrides", nargs="*", help="config overrides")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    if args.overrides:
        from omegaconf import OmegaConf
        cli_conf = OmegaConf.from_cli(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_conf)
    
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
    model = TriplaneGaussianAvatarLightning(cfg)
    
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
        model = TriplaneGaussianAvatarLightning.load_from_checkpoint(args.checkpoint, cfg=cfg)
        trainer.test(model, val_loader)
        
    elif args.mode == "predict":
        if args.checkpoint is None:
            raise ValueError("predict mode requires checkpoint file path")
        
        print(f"loading checkpoint: {args.checkpoint}")
        model = TriplaneGaussianAvatarLightning.load_from_checkpoint(args.checkpoint, cfg=cfg)
        trainer.predict(model, val_loader)
    
    print("program executed successfully!")

if __name__ == "__main__":
    main()
