#!/usr/bin/env python3
"""
测试新的堆叠集成模型
"""

import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.ensemble_model import EnsembleModel
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

def test_stacking_ensemble():
    """测试堆叠集成模型"""
    print("=== 测试新的堆叠集成模型 ===")
    
    # 生成测试数据
    print("生成测试数据...")
    X, y = make_regression(n_samples=1000, n_features=100, n_informative=50, 
                          n_targets=1, random_state=42, noise=0.1)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建堆叠集成模型
    print("\n创建堆叠集成模型...")
    ensemble = EnsembleModel(
        cv_folds=3,  # 使用较少的折数以加快测试
        task_type='regression',
        random_state=42
    )
    
    # 训练模型
    print("训练模型...")
    ensemble.train(X_train, y_train)
    
    # 预测
    print("进行预测...")
    y_pred = ensemble.predict(X_test)
    
    # 评估
    print("评估模型性能...")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    pcc, p_value = pearsonr(y_test, y_pred)
    
    print(f"测试集 R²: {r2:.4f}")
    print(f"测试集 MSE: {mse:.4f}")
    print(f"测试集 PCC: {pcc:.4f} (p={p_value:.4f})")
    
    # 获取模型信息
    print("\n获取模型信息...")
    model_info = ensemble.get_model_info()
    print(f"集成类型: {model_info['ensemble_type']}")
    print(f"交叉验证折数: {model_info['cv_folds']}")
    print(f"基础模型: {model_info['base_models']}")
    print(f"元模型: {model_info['meta_model']}")
    
    if 'meta_model_coefficients' in model_info:
        print("元模型系数:")
        for model_name, coef in model_info['meta_model_coefficients'].items():
            print(f"  {model_name}: {coef:.4f}")
    
    # 获取基础模型预测
    print("\n获取基础模型预测...")
    base_predictions = ensemble.get_base_model_predictions(X_test)
    for model_name, pred in base_predictions.items():
        model_r2 = r2_score(y_test, pred)
        model_pcc, _ = pearsonr(y_test, pred)
        print(f"{model_name}: R²={model_r2:.4f}, PCC={model_pcc:.4f}")
    
    # 获取特征重要性
    print("\n获取特征重要性...")
    try:
        feature_importance = ensemble.get_feature_importance()
        for model_name, importances in feature_importance.items():
            if importances:
                top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"{model_name} 前5个重要特征:")
                for feature, importance in top_features:
                    print(f"  {feature}: {importance:.4f}")
    except Exception as e:
        print(f"获取特征重要性时出错: {e}")
    
    print("\n=== 测试完成 ===")
    return ensemble

if __name__ == "__main__":
    test_stacking_ensemble() 