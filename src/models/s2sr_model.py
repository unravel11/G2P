import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from scipy.stats import pearsonr
from .base import BaseModel

class S2SRModel(BaseModel):
    def __init__(self, model_params=None, task_type='regression'):
        super().__init__(model_params)
        self.task_type = task_type
        self.ridge_alphas = model_params.get('ridge_alphas', [0.1, 1.0, 10.0, 100.0]) if model_params else [0.1, 1.0, 10.0, 100.0]
        self.lasso_cv = model_params.get('lasso_cv', True) if model_params else True
        self.pval_threshold = model_params.get('pval_threshold', 0.05) if model_params else 0.05
        self.max_group_size = model_params.get('max_group_size', 50) if model_params else 50
        self.model_main = None
        self.model_interaction = None
        self.selected_interaction_idx = None
        self.group_info = None
        self.feature_names = None
        self._init_model()

    def _init_model(self):
        """初始化模型实例"""
        # S2SR模型在训练时才创建具体的模型实例
        pass

    def fit(self, X, y, feature_names=None):
        """sklearn兼容的fit方法"""
        self.train(X, y, feature_names)
        return self

    def train(self, X, y, feature_names=None):
        self.feature_names = feature_names
        # 1. 主效应
        self.model_main = RidgeCV(alphas=self.ridge_alphas).fit(X, y)
        y_pred_main = self.model_main.predict(X)
        residuals = y - y_pred_main

        # 2. 分组与组均值
        X_group_mean, group_info = self._group_mean(X, feature_names)
        self.group_info = group_info
        X_interaction = self._build_interactions(X_group_mean)

        # 3. 统计筛选
        selected_idx = []
        for i in range(X_interaction.shape[1]):
            r, p = pearsonr(X_interaction[:, i], residuals)
            if p < self.pval_threshold:
                selected_idx.append(i)
        if not selected_idx:
            selected_idx = [np.argmin([pearsonr(X_interaction[:, i], residuals)[1] for i in range(X_interaction.shape[1])])]
        self.selected_interaction_idx = selected_idx
        X_interaction_filtered = X_interaction[:, selected_idx]

        # 4. Lasso拟合
        self.model_interaction = LassoCV().fit(X_interaction_filtered, residuals)

    def predict(self, X, feature_names=None):
        if feature_names is None:
            feature_names = self.feature_names
        y_pred_main = self.model_main.predict(X)
        X_group_mean, _ = self._group_mean(X, feature_names, self.group_info)
        X_interaction = self._build_interactions(X_group_mean)
        X_interaction_filtered = X_interaction[:, self.selected_interaction_idx]
        y_pred_residual = self.model_interaction.predict(X_interaction_filtered)
        return y_pred_main + y_pred_residual

    def get_params(self, deep=True):
        """获取模型参数，用于sklearn兼容性"""
        return {
            'ridge_alphas': self.ridge_alphas,
            'lasso_cv': self.lasso_cv,
            'pval_threshold': self.pval_threshold,
            'max_group_size': self.max_group_size,
            'task_type': self.task_type
        }

    def set_params(self, **params):
        """设置模型参数，用于sklearn兼容性"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _group_mean(self, X, feature_names, group_info=None):
        # 简单实现：每max_group_size个特征为一组，取均值
        n_features = X.shape[1]
        groups = []
        group_means = []
        if group_info is None:
            for i in range(0, n_features, self.max_group_size):
                idx = list(range(i, min(i+self.max_group_size, n_features)))
                groups.append(idx)
                group_means.append(X[:, idx].mean(axis=1))
            group_info = groups
        else:
            for idx in group_info:
                group_means.append(X[:, idx].mean(axis=1))
        X_group_mean = np.stack(group_means, axis=1)
        return X_group_mean, group_info

    def _build_interactions(self, X_group_mean):
        # 两两组均值交互项
        n_groups = X_group_mean.shape[1]
        interactions = []
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                interactions.append(X_group_mean[:, i] * X_group_mean[:, j])
        return np.stack(interactions, axis=1) if interactions else np.zeros((X_group_mean.shape[0], 0))

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model_main is None:
            return {}
        
        # 返回主效应模型的系数作为特征重要性
        importance = {}
        if hasattr(self.model_main, 'coef_'):
            coef = self.model_main.coef_
            for i, imp in enumerate(coef):
                if self.feature_names and i < len(self.feature_names):
                    importance[f"Main_{self.feature_names[i]}"] = abs(imp)
                else:
                    importance[f"Main_feature_{i}"] = abs(imp)
        
        return importance

    def get_model_summary(self):
        """获取模型摘要"""
        summary = {
            'model_type': 'S2SR',
            'main_model_alpha': self.model_main.alpha_ if hasattr(self.model_main, 'alpha_') else None,
            'main_model_coef_total': len(self.model_main.coef_) if hasattr(self.model_main, 'coef_') else 0,
            'main_model_coef_nonzero': np.sum(self.model_main.coef_ != 0) if hasattr(self.model_main, 'coef_') else 0,
            'interaction_features_total': len(self.selected_interaction_idx) if self.selected_interaction_idx else 0,
            'interaction_features_selected': len(self.selected_interaction_idx) if self.selected_interaction_idx else 0,
            'pval_threshold': self.pval_threshold,
            'max_group_size': self.max_group_size
        }
        return summary 