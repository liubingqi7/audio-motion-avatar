#!/usr/bin/env python3
"""
打印配置中的所有参数key
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.configs.config_loader import ConfigLoader
from omegaconf import OmegaConf

def print_config_keys(cfg, prefix=""):
    """
    递归打印配置中的所有key
    
    Args:
        cfg: 配置对象
        prefix: 当前路径前缀
    """
    try:
        if hasattr(cfg, 'keys'):
            for key in cfg.keys():
                current_path = f"{prefix}.{key}" if prefix else key
                value = cfg[key]
                if hasattr(value, 'keys'):
                    print(f"{current_path}: (dict)")
                    print_config_keys(value, current_path)
                elif isinstance(value, list):
                    print(f"{current_path}: (list) = {value}")
                else:
                    print(f"{current_path}: {type(value).__name__} = {value}")
    except Exception as e:
        print(f"Error processing {prefix}: {e}")

def main():
    """主函数"""
    print("=== 打印配置中的所有参数key ===")
    
    try:
        # 加载配置
        cfg = ConfigLoader.load_config("src/configs/config_thuman.yaml")
        
        print("✓ 配置加载成功")
        print(f"配置类型: {type(cfg)}")
        print(f"配置内容: {cfg}")
        
        print("\n配置结构:")
        print("=" * 50)
        
        # 打印所有key
        print_config_keys(cfg)
        
        print("\n" + "=" * 50)
        print("✓ 配置key打印完成")
        
    except Exception as e:
        print(f"✗ 打印配置key失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 