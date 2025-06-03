"""
模型评估工具
包含各种评估指标和交叉验证方法
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        Dict[str, float]: 包含各种评估指标的字典
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mse': mean_squared_error(y_true, y_pred)
    }
    
    logger.info("模型评估结果:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
        
    return metrics

def cross_validate(model: Any, X: np.ndarray, y: np.ndarray, 
                  n_splits: int = 5) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    进行交叉验证
    
    Args:
        model: 模型对象
        X: 特征矩阵
        y: 目标变量
        n_splits: 交叉验证折数
        
    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - 平均评估指标
            - 每折的评估指标列表
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        logger.info(f"开始第 {fold} 折交叉验证")
        
        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练模型
        model.train(X_train, y_train)
        
        # 预测并评估
        y_pred = model.predict(X_val)
        metrics = evaluate_model(y_val, y_pred)
        fold_metrics.append(metrics)
        
    # 计算平均指标
    mean_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    
    logger.info("\n交叉验证平均结果:")
    for metric_name, value in mean_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
        
    return mean_metrics, fold_metrics 