"""
基础模型类
定义所有预测模型的通用接口
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import joblib
import os

class BaseModel(ABC):
    """基础模型抽象类"""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        初始化模型
        
        Args:
            model_params: 模型参数字典
        """
        self.model_params = model_params or {}
        self.model = None
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测结果
        """
        pass
    
    def save(self, path: str) -> None:
        """
        保存模型到文件
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        joblib.dump(self.model, path)
        
    def load(self, path: str) -> None:
        """
        从文件加载模型
        
        Args:
            path: 模型文件路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        self.model = joblib.load(path) 