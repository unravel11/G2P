"""
随机森林模型实现
"""

from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from .base import BaseModel

class RandomForestModel(BaseModel):
    """随机森林模型类"""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, task_type: str = 'classification'):
        """
        初始化随机森林模型
        
        Args:
            model_params: 模型参数字典
            task_type: 任务类型，'classification' 或 'regression'
        """
        super().__init__(model_params)
        self.task_type = task_type
        self.feature_names = None  # 添加特征名称属性
        self._init_model()
        
    def _init_model(self) -> None:
        """初始化模型"""
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # 更新默认参数
        params = {**default_params, **self.model_params}
        
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称列表
        """
        self.feature_names = feature_names  # 保存特征名称
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测结果
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（仅用于分类任务）
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测概率
        """
        if self.task_type != 'classification':
            raise ValueError("概率预测仅适用于分类任务")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            Dict[str, float]: 评估指标
        """
        y_pred = self.predict(X)
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, y_pred)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        交叉验证
        
        Args:
            X: 特征矩阵
            y: 目标变量
            cv: 交叉验证折数
            
        Returns:
            Dict[str, float]: 交叉验证结果
        """
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            Dict[str, float]: 特征重要性字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        if self.feature_names is None:
            # 如果没有特征名称，使用索引作为特征名
            self.feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            
        return dict(zip(self.feature_names, self.model.feature_importances_)) 