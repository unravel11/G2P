"""
Ensemble参数合并工具
自动合并Ensemble和各基础模型的param_grid
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def merge_ensemble_param_grid(config: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    自动合并Ensemble和各基础模型的param_grid
    
    Args:
        config: 完整的配置字典
        
    Returns:
        Dict[str, List[Any]]: 合并后的param_grid
    """
    if 'Ensemble' not in config['models']:
        return {}
    
    # 获取Ensemble的param_grid
    ensemble_config = config['models']['Ensemble']
    ensemble_grid = ensemble_config.get('param_grid', {}).copy()
    
    # 获取models_config中定义的基础模型
    models_config = ensemble_config.get('models_config', {})
    
    # 模型名称映射（小写到标准名称）
    model_name_mapping = {
        'lasso': 'Lasso',
        'lightgbm': 'LightGBM', 
        'xgboost': 'XGBoost'
    }
    
    # 自动合并各基础模型的param_grid
    for model_name, model_cfg in models_config.items():
        # 查找对应的基础模型配置
        base_model_name = model_name_mapping.get(model_name, model_name.capitalize())
        if base_model_name in config['models']:
            base_model_config = config['models'][base_model_name]
            base_param_grid = base_model_config.get('param_grid', {})
            
            # 将基础模型的参数添加到Ensemble的param_grid中
            for param_name, param_values in base_param_grid.items():
                nested_param_name = f"{model_name}__{param_name}"
                ensemble_grid[nested_param_name] = param_values
                logger.info(f"自动添加参数: {nested_param_name} = {param_values}")
        else:
            logger.warning(f"未找到基础模型配置: {base_model_name}")
    
    return ensemble_grid

def get_ensemble_param_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    获取Ensemble参数的所有组合
    
    Args:
        config: 完整的配置字典
        
    Returns:
        List[Dict[str, Any]]: 所有参数组合的列表
    """
    from itertools import product
    
    merged_grid = merge_ensemble_param_grid(config)
    if not merged_grid:
        return []
    
    # 生成所有参数组合
    param_names = list(merged_grid.keys())
    param_values = list(merged_grid.values())
    
    combinations = []
    for values in product(*param_values):
        param_dict = dict(zip(param_names, values))
        combinations.append(param_dict)
    
    logger.info(f"生成 {len(combinations)} 个参数组合")
    return combinations

def print_ensemble_param_info(config: Dict[str, Any]) -> None:
    """
    打印Ensemble参数信息
    
    Args:
        config: 完整的配置字典
    """
    merged_grid = merge_ensemble_param_grid(config)
    if not merged_grid:
        print("未找到Ensemble配置")
        return
    
    print("\n=== Ensemble参数遍历信息 ===")
    print(f"总参数数量: {len(merged_grid)}")
    
    # 分类显示参数
    ensemble_params = {}
    base_model_params = {}
    
    for param_name, param_values in merged_grid.items():
        if '__' in param_name:
            base_model_params[param_name] = param_values
        else:
            ensemble_params[param_name] = param_values
    
    print("\nEnsemble参数:")
    for param_name, param_values in ensemble_params.items():
        print(f"  {param_name}: {param_values}")
    
    print("\n基础模型参数:")
    for param_name, param_values in base_model_params.items():
        print(f"  {param_name}: {param_values}")
    
    # 计算总组合数
    total_combinations = 1
    for param_values in merged_grid.values():
        total_combinations *= len(param_values)
    
    print(f"\n总参数组合数: {total_combinations}")
    print("=" * 30) 