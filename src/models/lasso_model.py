"""
Lasso回归模型实现
"""
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from .base import BaseModel

class LassoModel(BaseModel):
    """Lasso回归模型类"""
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, task_type: str = 'regression'):
        """
        初始化Lasso模型
        
        Args:
            model_params: 模型参数字典
            task_type: 任务类型，Lasso只支持回归任务
        """
        if task_type != 'regression':
            raise ValueError("Lasso模型只支持回归任务")
        super().__init__(model_params)
        self.feature_names = None
        self._init_model()

    def _init_model(self) -> None:
        default_params = {
            'alpha': 1.0,
            'random_state': 42
        }
        params = {**default_params, **self.model_params}
        self.model = Lasso(**params)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        self.feature_names = feature_names
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        return metrics

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("模型尚未训练")
        importances = np.abs(self.model.coef_)
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(len(importances))]
        return dict(zip(self.feature_names, importances)) 