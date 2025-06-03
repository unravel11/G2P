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
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
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

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("模型尚未训练")
        coefs = np.abs(self.model.coef_)
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(len(coefs))]
        return dict(zip(self.feature_names, coefs)) 