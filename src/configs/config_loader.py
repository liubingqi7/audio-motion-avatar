import os
import re
from omegaconf import OmegaConf, DictConfig

class ConfigLoader:
    """配置加载器，专注于处理变量引用和计算关系"""
    
    @staticmethod
    def load_config(config_path: str) -> DictConfig:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置对象
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 加载主配置文件
        cfg = OmegaConf.load(config_path)
        
        # 手动处理defaults
        if hasattr(cfg, 'defaults'):
            config_dir = os.path.dirname(config_path)
            merged_cfg = OmegaConf.create({})
            
            for default_config in cfg.defaults:
                if default_config == '_self_':
                    # 合并当前配置
                    merged_cfg = OmegaConf.merge(merged_cfg, cfg)
                else:
                    # 加载引用的配置文件
                    default_path = os.path.join(config_dir, f"{default_config}.yaml")
                    if os.path.exists(default_path):
                        default_cfg = OmegaConf.load(default_path)
                        merged_cfg = OmegaConf.merge(merged_cfg, default_cfg)
                    else:
                        print(f"警告: 配置文件不存在: {default_path}")
            
            cfg = merged_cfg
        
        # 处理变量引用和计算关系
        ConfigLoader._resolve_references(cfg)
        
        return cfg
    
    @staticmethod
    def _resolve_references(cfg: DictConfig):
        """
        处理配置中的变量引用和计算关系
        
        Args:
            cfg: 配置对象
        """
        # 递归处理所有配置项
        ConfigLoader._resolve_references_recursive(cfg, cfg)
    
    @staticmethod
    def _resolve_references_recursive(cfg: DictConfig, root_cfg: DictConfig):
        """
        递归处理变量引用和计算关系
        
        Args:
            cfg: 当前配置对象
            root_cfg: 根配置对象
        """
        # 先处理所有字符串值
        for key, value in cfg.items():
            if isinstance(value, str):
                # 处理字符串类型的变量引用和表达式
                resolved_value = ConfigLoader._resolve_string(value, root_cfg)
                cfg[key] = resolved_value
            elif isinstance(value, list):
                # 处理列表中的字符串
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        resolved_item = ConfigLoader._resolve_string(item, root_cfg)
                        value[i] = resolved_item
        
        # 然后递归处理嵌套配置
        for key, value in cfg.items():
            if isinstance(value, DictConfig):
                ConfigLoader._resolve_references_recursive(value, root_cfg)
    
    @staticmethod
    def _resolve_string(value: str, root_cfg: DictConfig):
        """
        解析字符串中的变量引用和表达式
        
        Args:
            value: 字符串值
            root_cfg: 根配置对象
            
        Returns:
            解析后的值
        """
        # 处理变量引用 ${...}
        if '${' in value and '}' in value:
            # 提取变量引用
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            
            for match in matches:
                # 解析变量路径
                resolved_var = ConfigLoader._get_value_by_path(match, root_cfg)
                if resolved_var is not None:
                    # 替换变量引用
                    value = value.replace(f'${{{match}}}', str(resolved_var))
            
            # 如果整个值就是一个变量引用，直接返回解析后的值
            if re.match(r'^\$\{[^}]+\}$', value):
                return ConfigLoader._get_value_by_path(matches[0], root_cfg)
        
        # 处理表达式（如 *3）
        if any(op in value for op in ['*', '+', '-', '/']):
            try:
                # 安全地评估表达式
                allowed_chars = set('0123456789.*+-/() ')
                if all(c in allowed_chars for c in value):
                    return eval(value)
            except:
                pass
        
        # 处理简单变量名（如 output_dim）
        if value in ['output_dim', 'input_dim']:
            return ConfigLoader._get_value_by_path(value, root_cfg)
        
        return value
    
    @staticmethod
    def _get_value_by_path(path: str, root_cfg: DictConfig):
        """
        根据路径获取配置值
        
        Args:
            path: 变量路径
            root_cfg: 根配置对象
            
        Returns:
            找到的值或None
        """
        try:
            # 如果是简单变量名，递归搜索
            if '.' not in path:
                return ConfigLoader._search_recursive(root_cfg, path)
            
            # 按路径分割
            path_parts = path.split('.')
            current = root_cfg
            
            for part in path_parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
            
            return current
        except:
            return None
    
    @staticmethod
    def _search_recursive(cfg: DictConfig, var_name: str):
        """
        递归搜索变量
        
        Args:
            cfg: 配置对象
            var_name: 变量名
            
        Returns:
            找到的值或None
        """
        # 在当前配置中查找
        if hasattr(cfg, var_name):
            return getattr(cfg, var_name)
        
        # 递归搜索所有嵌套配置
        for key, value in cfg.items():
            if isinstance(value, DictConfig):
                result = ConfigLoader._search_recursive(value, var_name)
                if result is not None:
                    return result
        
        return None
    
    @staticmethod
    def create_model_config(cfg: DictConfig) -> DictConfig:
        """
        创建模型配置对象
        
        Args:
            cfg: 配置对象
            
        Returns:
            模型配置对象
        """
        # 创建模型配置的副本
        model_cfg = OmegaConf.create({})
        
        # 提取各个模型组件的配置
        if hasattr(cfg, 'model'):
            # 提取ptv3_encoder配置
            if hasattr(cfg.model, 'ptv3_encoder'):
                for key, value in cfg.model.ptv3_encoder.items():
                    model_cfg[key] = value
            
            # 提取triplane_net配置
            if hasattr(cfg.model, 'triplane_net'):
                for key, value in cfg.model.triplane_net.items():
                    model_cfg[key] = value
            
            # 提取renderer配置
            if hasattr(cfg.model, 'renderer'):
                for key, value in cfg.model.renderer.items():
                    model_cfg[key] = value
            
            # 提取sapiens_encoder配置
            if hasattr(cfg.model, 'sapiens_encoder'):
                for key, value in cfg.model.sapiens_encoder.items():
                    model_cfg[key] = value
        
        # 提取训练配置到顶层（供模型使用）
        if hasattr(cfg, 'training'):
            for key, value in cfg.training.items():
                model_cfg[key] = value
        
        # 提取其他顶层配置
        for key in ['device', 'seed', 'experiment_name']:
            if hasattr(cfg, key):
                model_cfg[key] = cfg[key]
        
        return model_cfg 