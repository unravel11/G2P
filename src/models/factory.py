"""
模型工厂模块
用于动态创建模型实例
"""

from typing import Dict, Any, Type
from .base import BaseModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .lasso_model import LassoModel
from .cnn_model import CNNModel
from .gr2lm_model import GR2LMModel
from .s2sr_model import S2SRModel

class ModelFactory:
    """模型工厂类"""
    
    # 模型类映射表
    _model_classes: Dict[str, Type[BaseModel]] = {
        'RandomForestModel': RandomForestModel,
        'XGBoostModel': XGBoostModel,
        'LightGBMModel': LightGBMModel,
        'LassoModel': LassoModel,
        'CNNModel': CNNModel,
        'GR2LMModel': GR2LMModel,
        'S2SRModel': S2SRModel
    }
    
    @classmethod
    def create_model(cls, model_name: str, model_config: Dict[str, Any]) -> BaseModel:
        """
        创建模型实例
        
        Args:
            model_name: 模型名称
            model_config: 模型配置字典
            
        Returns:
            BaseModel: 模型实例
            
        Raises:
            ValueError: 如果模型名称无效
        """
        model_class_name = model_config['class']
        if model_class_name not in cls._model_classes:
            raise ValueError(f"未知的模型类: {model_class_name}")
            
        model_class = cls._model_classes[model_class_name]
        model_params = model_config.get('params', {})
        
        # 如果模型配置中包含task_type，则添加到参数字典中
        if 'task_type' in model_config:
            return model_class(model_params=model_params, task_type=model_config['task_type'])
        else:
            return model_class(model_params=model_params)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        注册新的模型类
        
        Args:
            name: 模型类名称
            model_class: 模型类
        """
        cls._model_classes[name] = model_class 