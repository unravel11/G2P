"""
模型训练工具
包含通用的模型训练和评估函数
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from utils.evaluation import evaluate_model, plot_prediction_vs_actual
from utils.hyperparameter_tuning import grid_search
import os

logger = logging.getLogger(__name__)

def train_and_evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_jobs: int = -1,
    output_dir: Optional[str] = None,
    prediction_dir: Optional[str] = None,
    prefix: Optional[str] = None
) -> Dict[str, Any]:
    """
    训练和评估模型
    
    Args:
        model: 模型对象
        X_train: 训练集特征矩阵
        y_train: 训练集目标变量
        X_val: 验证集特征矩阵
        y_val: 验证集目标变量
        X_test: 测试集特征矩阵
        y_test: 测试集目标变量
        feature_names: 特征名称列表
        param_grid: 参数网格
        n_jobs: 并行计算的CPU核心数
        output_dir: 输出目录（用于保存loss曲线）
        prediction_dir: 预测图保存目录
        prefix: 文件名前缀
        
    Returns:
        Dict[str, Any]: 包含模型和评估结果的字典
    """
    from utils.hyperparameter_tuning import grid_search
    
    # 如果是CNN模型,需要特殊处理
    is_cnn = model.__class__.__name__ == 'CNNModel'
    
    # 如果有参数网格，进行参数搜索
    if param_grid is not None:
        logger.info("开始参数搜索...")
        if is_cnn:
            # CNN模型使用自定义的参数搜索
            search_results = grid_search(
                model=model,
                X=X_train,
                y=y_train,
                param_grid=param_grid,
                n_jobs=1  # CNN模型不支持并行训练
            )
        else:
            search_results = grid_search(
                model=model,
                X=X_train,
                y=y_train,
                param_grid=param_grid,
                n_jobs=n_jobs
            )
        model.model = search_results['best_estimator']
        param_results = search_results['all_results']
    else:
        param_results = None
    
    # 训练模型
    logger.info("训练最终模型...")
    if is_cnn:
        # 为CNN模型创建loss_curve目录
        if output_dir:
            # 直接使用output_dir作为基础路径
            loss_curve_dir = output_dir  # 直接使用传入的目录
            os.makedirs(loss_curve_dir, exist_ok=True)
            logger.info(f"loss曲线将保存到: {loss_curve_dir}")
        else:
            loss_curve_dir = None
            logger.warning("未提供output_dir，loss曲线将不会保存")
            
        model.train(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names, output_dir=loss_curve_dir)
    else:
        model.train(X_train, y_train, feature_names=feature_names)
    
    # 在测试集上评估
    logger.info("在测试集上评估模型...")
    y_pred = model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_pred)
    
    # 绘制预测图
    if prediction_dir and prefix:
        os.makedirs(prediction_dir, exist_ok=True)
        plot_prediction_vs_actual(y_test, y_pred, output_dir=prediction_dir, prefix=prefix)
    
    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    top_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'top_features': top_features,
        'param_results': param_results
    } 