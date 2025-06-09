"""
超参数搜索工具
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from sklearn.model_selection import GridSearchCV
from models.base import BaseModel

logger = logging.getLogger(__name__)

def grid_search(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int = 5,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    执行网格搜索
    
    Args:
        model: 模型实例
        X: 特征矩阵
        y: 目标变量
        param_grid: 参数网格
        cv: 交叉验证折数
        n_jobs: 并行计算的CPU核心数
        
    Returns:
        dict: 包含最佳参数和所有参数组合结果的字典
    """
    logger.info("开始网格搜索...")
    logger.info(f"参数网格: {param_grid}")
    
    # 创建网格搜索对象
    grid = GridSearchCV(
        estimator=model.model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=n_jobs,
        return_train_score=True  # 返回训练集得分
    )
    
    # 执行网格搜索
    grid.fit(X, y)
    
    # 收集所有参数组合的结果
    all_results = []
    for i, params in enumerate(grid.cv_results_['params']):
        result = {
            'params': params,
            'mean_test_score': grid.cv_results_['mean_test_score'][i],
            'std_test_score': grid.cv_results_['std_test_score'][i],
            'mean_train_score': grid.cv_results_['mean_train_score'][i],
            'std_train_score': grid.cv_results_['std_train_score'][i],
            'rank_test_score': grid.cv_results_['rank_test_score'][i]
        }
        all_results.append(result)
    
    # 按测试集得分排序
    all_results.sort(key=lambda x: x['mean_test_score'], reverse=True)
    
    logger.info("网格搜索完成:")
    logger.info(f"最佳参数: {grid.best_params_}")
    logger.info(f"最佳得分: {grid.best_score_:.4f}")
    
    # 返回所有结果
    return {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'all_results': all_results,
        'best_estimator': grid.best_estimator_
    } 