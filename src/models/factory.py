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
from .ensemble_model import EnsembleModel
from copy import deepcopy

class ModelFactory:
    """模型工厂类"""
    
    # 模型类映射表
    _model_classes: Dict[str, Type[BaseModel]] = {
        'RandomForestModel': RandomForestModel,
        'XGBoostModel': XGBoostModel,
        'LightGBMModel': LightGBMModel,
        'LassoModel': LassoModel,
        'CNNModel': CNNModel,
        'EnsembleModel': EnsembleModel
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
        
        # 处理EnsembleModel的复杂参数
        if model_class_name == 'EnsembleModel':
            # 1. 拷贝原始models_config
            models_config = deepcopy(model_config.get('models_config', {}))
            # 2. 处理params
            params = model_config.get('params', {})
            ensemble_params = {}
            for k, v in params.items():
                if '__' in k:
                    # 处理嵌套参数，如 models_config__elasticnet__alphas
                    parts = k.split('__')
                    if len(parts) >= 3 and parts[0] == 'models_config':
                        model_name = parts[1]
                        param_name = '__'.join(parts[2:])
                        if model_name in models_config:
                            models_config[model_name][param_name] = v
                    else:
                        ensemble_params[k] = v
                else:
                    ensemble_params[k] = v
            # 3. 组装参数
            cv = ensemble_params.get('cv', model_config.get('cv', 5))
            task_type = model_config.get('task_type', 'regression')
            random_state = ensemble_params.get('random_state', model_config.get('random_state', 42))
            
            return model_class(
                models_config=models_config,
                cv=cv,
                task_type=task_type,
                random_state=random_state
            )
        else:
            # 其他模型的常规处理
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