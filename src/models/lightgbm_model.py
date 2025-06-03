"""
LightGBM模型实现
"""
from typing import Dict, Any, Optional, List
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from .base import BaseModel

class LightGBMModel(BaseModel):
    """LightGBM模型类"""
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, task_type: str = 'classification'):
        super().__init__(model_params)
        self.task_type = task_type
        self.feature_names = None
        self._init_model()

    def _init_model(self) -> None:
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'random_state': 42
        }
        params = {**default_params, **self.model_params}
        if self.task_type == 'classification':
            self.model = LGBMClassifier(**params)
        else:
            self.model = LGBMRegressor(**params)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        self.feature_names = feature_names
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type != 'classification':
            raise ValueError("概率预测仅适用于分类任务")
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
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
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("模型尚未训练")
        importances = self.model.feature_importances_
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(len(importances))]
        return dict(zip(self.feature_names, importances)) 