"""
堆叠集成模型实现
基于scikit-learn的StackingRegressor，防止数据泄露
"""
from typing import Dict, Any, Optional, List, Union
import numpy as np
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from .base import BaseModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel

class EnsembleModel(BaseModel):
    """堆叠集成模型类，基于StackingRegressor实现"""
    
    def __init__(self, 
                 models_config: Optional[Dict[str, Dict[str, Any]]] = None,
                 cv: int = 5,
                 task_type: str = 'regression',
                 random_state: int = 42):
        """
        初始化堆叠集成模型
        
        Args:
            models_config: 各个模型的配置字典
            cv: 交叉验证折数，用于防止数据泄露
            task_type: 任务类型 ('regression' 或 'classification')
            random_state: 随机种子
        """
        self.cv = cv
        self.task_type = task_type
        self.random_state = random_state
        self.feature_names = None
        
        # 默认模型配置
        default_config = {
            'elasticnet': {
                'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
                'alphas': [0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': 2000,
                'random_state': random_state
            },
            'svr': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 10,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': random_state,
                'verbose': -1
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': random_state
            },
            'meta_model': {
                'alphas': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        self.models_config = {**default_config, **(models_config or {})}
        
        # 调用父类初始化
        super().__init__()
        
        # 初始化模型
        self._init_model()

    def _init_model(self) -> None:
        """初始化堆叠集成模型"""
        # 创建基础模型
        base_models = []
        
        # 1. ElasticNetCV - 正则化线性模型
        elasticnet_params = self.models_config.get('elasticnet', {})
        elasticnet = ElasticNetCV(**elasticnet_params)
        base_models.append(('elasticnet', elasticnet))
        
        # 2. SVR - 核方法
        svr_params = self.models_config.get('svr', {})
        svr = SVR(**svr_params)
        base_models.append(('svr', svr))
        
        # 3. LightGBM - 树的集成模型
        lgbm_params = self.models_config.get('lightgbm', {})
        lgbm_model = LightGBMModel(model_params=lgbm_params, task_type=self.task_type)
        base_models.append(('lightgbm', lgbm_model.model))
        
        # 4. XGBoost - 树的集成模型
        xgb_params = self.models_config.get('xgboost', {})
        xgb_model = XGBoostModel(model_params=xgb_params, task_type=self.task_type)
        base_models.append(('xgboost', xgb_model.model))
        
        # 创建元模型
        meta_params = self.models_config.get('meta_model', {})
        if self.task_type == 'regression':
            meta_model = RidgeCV(**meta_params)
            self.model = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_model,
                cv=self.cv,
                n_jobs=-1,
                passthrough=False  # 不使用原始特征，只使用基础模型预测
            )
        else:
            meta_model = RidgeCV(**meta_params)
            self.model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=self.cv,
                n_jobs=-1,
                passthrough=False
            )
        
        # 保存基础模型引用，用于特征重要性分析
        self.base_models = {
            'elasticnet': elasticnet,
            'svr': svr,
            'lightgbm': lgbm_model,
            'xgboost': xgb_model
        }

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        训练堆叠集成模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称列表
        """
        self.feature_names = feature_names
        
        # 使用StackingRegressor/StackingClassifier的fit方法
        # 它会自动处理交叉验证和元模型训练
        self.model.fit(X, y)
        
        print(f"[EnsembleModel][train] 堆叠集成模型训练完成")
        print(f"[EnsembleModel][train] 基础模型数量: {len(self.model.estimators_)}")
        print(f"[EnsembleModel][train] 元模型: {type(self.model.final_estimator_).__name__}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用堆叠集成模型进行预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测结果
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（仅适用于分类任务）
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测概率
        """
        if self.task_type != 'classification':
            raise ValueError("概率预测仅适用于分类任务")
        
        return self.model.predict_proba(X)

    def get_base_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        获取各个基础模型的预测结果
        
        Args:
            X: 特征矩阵
            
        Returns:
            Dict[str, np.ndarray]: 各基础模型的预测结果
        """
        predictions = {}
        base_model_names = [name for name, _ in self.model.estimators]
        for idx, estimator in enumerate(self.model.estimators_):
            name = base_model_names[idx]
            if hasattr(estimator, 'predict'):
                predictions[name] = estimator.predict(X)
            else:
                predictions[name] = estimator.transform(X)
        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估堆叠集成模型性能
        
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
        
        # 添加各个基础模型的单独评估结果
        base_predictions = self.get_base_model_predictions(X)
        for model_name, pred in base_predictions.items():
            if self.task_type == 'classification':
                model_metrics = {
                    'accuracy': accuracy_score(y, pred)
                }
            else:
                model_metrics = {
                    'mse': mean_squared_error(y, pred),
                    'r2': r2_score(y, pred)
                }
            
            for metric_name, value in model_metrics.items():
                metrics[f'{model_name}_{metric_name}'] = value
        
        return metrics

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        获取特征重要性
        
        Returns:
            Dict[str, Dict[str, float]]: 各模型的特征重要性
        """
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("模型尚未训练")
        
        importance_dict = {}
        base_model_names = [name for name, _ in self.model.estimators]
        
        # 获取各基础模型的特征重要性
        for idx, estimator in enumerate(self.model.estimators_):
            name = base_model_names[idx]
            try:
                if hasattr(estimator, 'feature_importances_'):
                    # 树模型
                    importances = estimator.feature_importances_
                elif hasattr(estimator, 'coef_'):
                    # 线性模型
                    importances = np.abs(estimator.coef_)
                else:
                    # 不支持特征重要性的模型
                    importances = np.zeros(len(self.feature_names)) if self.feature_names else np.array([])
                
                if self.feature_names and len(importances) == len(self.feature_names):
                    importance_dict[name] = dict(zip(self.feature_names, importances))
                else:
                    importance_dict[name] = {f"feature_{i}": imp for i, imp in enumerate(importances)}
                    
            except Exception as e:
                print(f"获取{name}模型特征重要性时出错: {e}")
                importance_dict[name] = {}
        
        # 添加元模型的特征重要性（基础模型权重）
        try:
            if hasattr(self.model.final_estimator_, 'coef_'):
                meta_importances = np.abs(self.model.final_estimator_.coef_.flatten())
                importance_dict['meta_model'] = dict(zip(base_model_names, meta_importances))
        except Exception as e:
            print(f"获取元模型特征重要性时出错: {e}")
        
        return importance_dict

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        info = {
            'ensemble_type': 'stacking',
            'cv': self.cv,
            'task_type': self.task_type,
            'base_models': [name for name, _ in self.model.estimators],
            'meta_model': type(self.model.final_estimator_).__name__
        }
        
        # 添加元模型的系数信息
        if hasattr(self.model.final_estimator_, 'coef_'):
            base_model_names = [name for name, _ in self.model.estimators]
            coefficients = self.model.final_estimator_.coef_.flatten()
            info['meta_model_coefficients'] = dict(zip(base_model_names, coefficients))
        
        return info

    def get_cv_scores(self) -> Dict[str, float]:
        """
        获取交叉验证分数
        
        Returns:
            Dict[str, float]: 交叉验证分数
        """
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("模型尚未训练")
        
        scores = {}
        
        # 获取各基础模型的交叉验证分数
        for name, estimator in self.model.estimators_:
            try:
                if hasattr(estimator, 'cv_scores_'):
                    scores[f'{name}_cv_score'] = np.mean(estimator.cv_scores_)
                elif hasattr(estimator, 'best_score_'):
                    scores[f'{name}_best_score'] = estimator.best_score_
            except Exception as e:
                print(f"获取{name}模型交叉验证分数时出错: {e}")
        
        return scores 