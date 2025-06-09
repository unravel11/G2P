"""
模型训练工具
包含通用的模型训练和评估函数
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.base import BaseModel
from utils.evaluation import evaluate_model, cross_validate, plot_prediction_vs_actual
from utils.hyperparameter_tuning import grid_search

logger = logging.getLogger(__name__)

def train_and_evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_jobs: int = -1,
    cv: int = 3,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None
) -> Dict[str, Any]:
    """
    训练和评估模型
    
    Args:
        model: 模型对象
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        test_size: 测试集比例
        random_state: 随机种子
        n_splits: 交叉验证折数
        param_grid: 参数网格
        n_jobs: 并行计算的CPU核心数
        cv: 参数搜索的交叉验证折数
        output_dir: 输出目录
        prefix: 文件名前缀
        
    Returns:
        Dict[str, Any]: 包含模型和评估结果的字典
    """
    from sklearn.model_selection import train_test_split
    from utils.hyperparameter_tuning import grid_search
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 如果有参数网格，进行参数搜索
    if param_grid is not None:
        logger.info("开始参数搜索...")
        search_results = grid_search(
            model=model,
            X=X_train,
            y=y_train,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs
        )
        model.model = search_results['best_estimator']
        param_results = search_results['all_results']
    else:
        param_results = None
    
    # 训练模型
    logger.info("训练最终模型...")
    model.train(X_train, y_train, feature_names=feature_names)
    
    # 在测试集上评估
    logger.info("在测试集上评估模型...")
    y_pred = model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_pred)
    
    # 绘制预测图
    if output_dir and prefix:
        plot_prediction_vs_actual(y_test, y_pred, output_dir=output_dir, prefix=prefix)
    
    # 进行交叉验证
    logger.info("进行交叉验证...")
    cv_metrics = cross_validate(model, X, y, n_splits=n_splits)
    
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
        'cv_metrics': cv_metrics,
        'top_features': top_features,
        'param_results': param_results
    } 