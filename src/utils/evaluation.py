"""
模型评估工具
包含各种评估指标和交叉验证方法
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Any, List, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

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
    # 计算皮尔逊相关系数
    pearson_r, p_value = pearsonr(y_true, y_pred)
    
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mse': mean_squared_error(y_true, y_pred),
        'pearson_r': pearson_r,
        'pearson_p': p_value
    }
    
    logger.info("\n测试集评估结果:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
        
    return metrics

def cross_validate(model: Any, X: np.ndarray, y: np.ndarray, n_splits: int = 5, is_cnn: bool = False) -> Dict[str, float]:
    """
    进行交叉验证
    
    Args:
        model: 模型对象
        X: 特征矩阵
        y: 目标变量
        n_splits: 交叉验证折数
        is_cnn: 是否是CNN模型
        
    Returns:
        Dict[str, float]: 交叉验证结果
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    logger.info(f"\n开始 {n_splits} 折交叉验证:")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        logger.info(f"\n第 {fold} 折:")
        
        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练模型
        if is_cnn:
            # CNN模型需要验证集
            X_train, X_val_inner, y_train, y_val_inner = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            model.train(X_train, y_train, X_val=X_val_inner, y_val=y_val_inner)
        else:
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
        
    return mean_metrics

def plot_feature_importance(
    feature_importance: List[Tuple[str, float]],
    top_n: int = 10,
    output_dir: str = None,
    prefix: str = None
) -> None:
    """
    绘制特征重要性图
    
    Args:
        feature_importance: 特征重要性列表，每个元素为(特征名, 重要性值)的元组
        top_n: 显示前N个重要特征
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    # 设置绘图风格
    plt.style.use('seaborn')
    
    # 获取前N个重要特征
    top_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:top_n]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建水平条形图
    bars = plt.barh(
        [x[0] for x in top_features],
        [x[1] for x in top_features],
        color='#2878B5',
        alpha=0.8
    )
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center',
                fontsize=10, fontweight='bold')
    
    # 设置标题和标签
    plt.title('Feature Importance', fontsize=14, pad=15)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{prefix}_feature_importance.png" if prefix else "feature_importance.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str = None, prefix: str = None):
    """
    绘制预测值vs实际值散点图
    
    Args:
        y_true: 真实值（测试集）
        y_pred: 预测值（测试集）
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    # 设置绘图风格
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    plt.scatter(y_true, y_pred, alpha=0.6, c='#2878B5', s=50)
    
    # 绘制对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置标签和标题
    plt.xlabel('Actual Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.title('Predicted vs Actual Values', fontsize=14, pad=15)
    
    # 计算评估指标
    pearson_r, p_value = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 添加评估指标文本
    metrics_text = (
        f'R² = {r2:.3f}\n'
        f'PCC = {pearson_r:.3f}\n'
        f'p-value = {p_value:.2e}\n'
        f'RMSE = {rmse:.3f}'
    )
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
             fontsize=10, verticalalignment='top')
    
    # 添加图例
    plt.legend(loc='lower right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = 'prediction_vs_actual.png'
    if prefix:
        filename = f"{prefix}_{filename}"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close() 