"""
超参数搜索工具
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from models.base import BaseModel
from models.cnn_model import CNNModel
from models.ensemble_model import EnsembleModel
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from .ensemble_param_merger import merge_ensemble_param_grid

logger = logging.getLogger(__name__)

def pearson_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

pcc_scorer = make_scorer(pearson_scorer, greater_is_better=True)

def grid_search(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int = 5,
    n_jobs: int = -1,
    config: Dict[str, Any] = None
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
        config: 完整配置字典（用于Ensemble参数合并）
        
    Returns:
        dict: 包含最佳参数和所有参数组合结果的字典
    """
    logger.info("开始网格搜索...")
    
    # 如果是EnsembleModel，自动合并参数
    if isinstance(model, EnsembleModel) and config:
        merged_param_grid = merge_ensemble_param_grid(config)
        if merged_param_grid:
            logger.info("检测到Ensemble模型，自动合并基础模型参数")
            logger.info(f"原始参数网格: {param_grid}")
            logger.info(f"合并后参数网格: {merged_param_grid}")
            param_grid = merged_param_grid
    
    logger.info(f"最终参数网格: {param_grid}")
    
    # 如果是CNNModel，手写参数搜索
    if isinstance(model, CNNModel):
        all_results = []
        best_score = -np.inf
        best_params = None
        best_model = None
        param_list = list(ParameterGrid(param_grid))
        logger.info(f"CNN模型参数组合数: {len(param_list)}")
        for i, params in enumerate(param_list):
            logger.info(f"[{i+1}/{len(param_list)}] 当前参数: {params}")
            # 创建新模型实例
            cnn = CNNModel(model_params=params, task_type=model.task_type)
            # 划分训练/验证集
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            try:
                cnn.train(X_train, y_train, X_val=X_val, y_val=y_val)
                metrics = cnn.evaluate(X_val, y_val)
                # 用pearson_r（pcc）作为主评判标准
                score = metrics.get('pearson_r', -np.inf)
                logger.info(f"参数: {params}, 验证集PCC: {score:.4f}, R2: {metrics.get('r2', float('nan')):.4f}")
                all_results.append({'params': params, 'val_score': score, 'metrics': metrics})
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = cnn
            except Exception as e:
                logger.error(f"参数: {params} 训练/评估出错: {e}")
                all_results.append({'params': params, 'val_score': -np.inf, 'metrics': {}, 'error': str(e)})
        logger.info(f"CNN参数搜索完成，最佳参数: {best_params}, 最佳R2: {best_score:.4f}")
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'best_estimator': best_model
        }
    
    # 如果是EnsembleModel，手写参数搜索
    if isinstance(model, EnsembleModel):
        all_results = []
        best_score = -np.inf
        best_params = None
        best_model = None
        param_list = list(ParameterGrid(param_grid))
        logger.info(f"Ensemble模型参数组合数: {len(param_list)}")
        for i, params in enumerate(param_list):
            logger.info(f"[{i+1}/{len(param_list)}] 当前参数: {params}")
            # 创建新模型实例
            from models.factory import ModelFactory
            ensemble_config = config['models']['Ensemble'].copy()
            ensemble_config['params'] = params
            ensemble = ModelFactory.create_model('Ensemble', ensemble_config)
            # 划分训练/验证集
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            try:
                ensemble.train(X_train, y_train)
                metrics = ensemble.evaluate(X_val, y_val)
                # 用pearson_r（pcc）作为主评判标准
                score = metrics.get('pearson_r', -np.inf)
                logger.info(f"参数: {params}, 验证集PCC: {score:.4f}, R2: {metrics.get('r2', float('nan')):.4f}")
                all_results.append({'params': params, 'val_score': score, 'metrics': metrics})
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = ensemble
            except Exception as e:
                logger.error(f"参数: {params} 训练/评估出错: {e}")
                all_results.append({'params': params, 'val_score': -np.inf, 'metrics': {}, 'error': str(e)})
        logger.info(f"Ensemble参数搜索完成，最佳参数: {best_params}, 最佳PCC: {best_score:.4f}")
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'best_estimator': best_model
        }
    
    # 其它模型，使用GridSearchCV（包括新的堆叠集成模型）
    grid = GridSearchCV(
        estimator=model.model,
        param_grid=param_grid,
        cv=cv,
        scoring=pcc_scorer,
        n_jobs=n_jobs,
        return_train_score=True
    )
    grid.fit(X, y)
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
    all_results.sort(key=lambda x: x['mean_test_score'], reverse=True)
    logger.info("网格搜索完成:")
    logger.info(f"最佳参数: {grid.best_params_}")
    logger.info(f"最佳得分: {grid.best_score_:.4f}")
    return {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'all_results': all_results,
        'best_estimator': grid.best_estimator_
    } 