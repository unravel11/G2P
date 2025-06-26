"""
分组正则化与残差修正线性模型 (Group-Regularized and Residual-Correcting Linear Model, GR²-LM)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from .base import BaseModel
import warnings
warnings.filterwarnings('ignore')

class GR2LMModel(BaseModel, BaseEstimator, RegressorMixin):
    """
    分组正则化与残差修正线性模型
    
    该模型由两个协同工作的阶段构成：
    1. 主效应建模：使用ElasticNet捕捉SNP的主要线性效应
    2. 残差修正建模：使用基于分组交互的Lasso修正非线性相互作用效应
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, task_type: str = 'regression', **kwargs):
        """
        初始化GR²-LM模型
        
        Args:
            model_params: 模型参数字典，包含以下参数：
                - main_alpha: ElasticNet的alpha参数
                - main_l1_ratio: ElasticNet的l1_ratio参数
                - residual_alpha: 残差修正Lasso的alpha参数
                - ld_threshold: 连锁不平衡阈值，用于SNP分组
                - min_group_size: 最小组大小
                - max_group_size: 最大组大小
            task_type: 任务类型，目前只支持回归任务
            **kwargs: 其他参数，用于sklearn兼容性
        """
        if task_type != 'regression':
            raise ValueError("GR²-LM模型目前只支持回归任务")
        
        # 处理sklearn GridSearchCV传递的参数
        if model_params is None:
            model_params = {}
        
        # 将kwargs中的参数合并到model_params中
        for key, value in kwargs.items():
            model_params[key] = value
        
        # 初始化sklearn接口
        self.model_params = model_params
        self.task_type = task_type
        self.feature_names = None
        self.snp_groups = None
        self.group_representatives = None
        self.scaler = StandardScaler()
        
        # 初始化BaseModel
        super().__init__(model_params)
        self._init_model()
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'GR2LMModel':
        """
        sklearn兼容的fit方法
        
        Args:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称列表
            
        Returns:
            self: 训练后的模型实例
        """
        self.train(X, y, feature_names)
        return self
    
    def _init_model(self) -> None:
        """初始化模型参数"""
        default_params = {
            'main_alpha': 0.1,           # ElasticNet的alpha参数
            'main_l1_ratio': 0.5,        # ElasticNet的l1_ratio参数
            'residual_alpha': 0.01,      # 残差修正Lasso的alpha参数
            'ld_threshold': 0.5,         # 连锁不平衡阈值
            'min_group_size': 2,         # 最小组大小
            'max_group_size': 10,        # 最大组大小
            'random_state': 42
        }
        
        # 更新默认参数
        for key, value in self.model_params.items():
            if key in default_params:
                default_params[key] = value
        
        self.model_params = default_params
        
        # 初始化模型组件
        self.main_model = None
        self.residual_model = None
        self.is_trained = False
    
    def _calculate_ld_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        计算连锁不平衡矩阵
        
        Args:
            X: SNP数据矩阵 (samples x SNPs)
            
        Returns:
            np.ndarray: 连锁不平衡矩阵
        """
        # 计算相关系数矩阵作为LD矩阵的近似
        ld_matrix = np.corrcoef(X.T)
        # 将对角线设为0，避免自相关
        np.fill_diagonal(ld_matrix, 0)
        return np.abs(ld_matrix)
    
    def _group_snps_by_ld(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> List[List[int]]:
        """
        基于连锁不平衡对SNP进行分组
        
        Args:
            X: SNP数据矩阵
            feature_names: 特征名称列表
            
        Returns:
            List[List[int]]: SNP分组列表
        """
        ld_matrix = self._calculate_ld_matrix(X)
        n_snps = X.shape[1]
        
        # 初始化分组
        groups = []
        used_snps = set()
        
        for i in range(n_snps):
            if i in used_snps:
                continue
            
            # 创建新组
            current_group = [i]
            used_snps.add(i)
            
            # 寻找与当前SNP高度相关的其他SNP
            for j in range(i + 1, n_snps):
                if j in used_snps:
                    continue
                
                # 检查LD值是否超过阈值
                if ld_matrix[i, j] >= self.model_params['ld_threshold']:
                    # 检查组大小限制
                    if len(current_group) < self.model_params['max_group_size']:
                        current_group.append(j)
                        used_snps.add(j)
            
            # 如果组太小，将单个SNP作为独立组
            if len(current_group) >= self.model_params['min_group_size']:
                groups.append(current_group)
            else:
                # 将单个SNP作为独立组
                groups.append([i])
        
        return groups
    
    def _extract_group_representatives(self, X: np.ndarray, groups: List[List[int]]) -> Tuple[np.ndarray, List[str]]:
        """
        提取每个组的代表特征（使用多种策略）
        
        Args:
            X: SNP数据矩阵
            groups: SNP分组列表
            
        Returns:
            Tuple[np.ndarray, List[str]]: 组代表特征矩阵和特征名称
        """
        group_features = []
        group_names = []
        
        for i, group in enumerate(groups):
            if len(group) == 1:
                # 单个SNP，直接使用
                group_features.append(X[:, group[0]])
                group_names.append(f"Group_{i}_SNP_{group[0]}")
            else:
                # 多个SNP，使用多种策略提取代表特征
                group_data = X[:, group]
                
                # 策略1：使用PCA提取前2个主成分
                if group_data.shape[1] >= 2:
                    n_components = min(2, group_data.shape[1])
                    pca = PCA(n_components=n_components, random_state=self.model_params['random_state'])
                    pcs = pca.fit_transform(group_data)
                    
                    for j in range(n_components):
                        group_features.append(pcs[:, j])
                        group_names.append(f"Group_{i}_PC{j+1}")
                else:
                    # 如果SNP数量不足，使用单个PC
                    pca = PCA(n_components=1, random_state=self.model_params['random_state'])
                    pc1 = pca.fit_transform(group_data).flatten()
                    group_features.append(pc1)
                    group_names.append(f"Group_{i}_PC1")
                
                # 策略2：添加组内SNP的均值作为额外特征
                group_mean = np.mean(group_data, axis=1)
                group_features.append(group_mean)
                group_names.append(f"Group_{i}_Mean")
        
        return np.column_stack(group_features), group_names
    
    def _create_interaction_features(self, group_features: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        创建组间交互特征
        
        Args:
            group_features: 组代表特征矩阵
            
        Returns:
            Tuple[np.ndarray, List[str]]: 交互特征矩阵和特征名称
        """
        n_groups = group_features.shape[1]
        interaction_features = []
        interaction_names = []
        
        # 创建两两交互特征
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                interaction = group_features[:, i] * group_features[:, j]
                interaction_features.append(interaction)
                interaction_names.append(f"Interaction_{i}_{j}")
        
        if interaction_features:
            return np.column_stack(interaction_features), interaction_names
        else:
            return np.array([]).reshape(group_features.shape[0], 0), []
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        训练GR²-LM模型
        
        Args:
            X: 特征矩阵 (SNP数据)
            y: 目标变量 (表型数据)
            feature_names: 特征名称列表
        """
        self.feature_names = feature_names
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 第一阶段：主效应建模
        print("第一阶段：主效应建模...")
        self.main_model = ElasticNet(
            alpha=self.model_params['main_alpha'],
            l1_ratio=self.model_params['main_l1_ratio'],
            random_state=self.model_params['random_state']
        )
        self.main_model.fit(X_scaled, y)
        
        # 计算主效应预测和残差
        y_pred_main = self.main_model.predict(X_scaled)
        residuals = y - y_pred_main
        
        # 计算主效应PCC
        main_pcc = np.corrcoef(y, y_pred_main)[0, 1]
        print(f"主效应模型PCC: {main_pcc:.4f}")
        print(f"残差方差: {np.var(residuals):.4f}")
        
        # 第二阶段：残差修正建模
        print("第二阶段：残差修正建模...")
        
        # SNP分组
        self.snp_groups = self._group_snps_by_ld(X_scaled)
        print(f"SNP分组完成，共{len(self.snp_groups)}个组")
        
        # 提取组代表特征
        group_features, group_names = self._extract_group_representatives(X_scaled, self.snp_groups)
        self.group_representatives = group_names
        
        # 创建交互特征
        interaction_features, interaction_names = self._create_interaction_features(group_features)
        
        if interaction_features.size > 0:
            print(f"创建了{len(interaction_names)}个交互特征")
            
            # 训练残差修正模型
            self.residual_model = Lasso(
                alpha=self.model_params['residual_alpha'],
                random_state=self.model_params['random_state']
            )
            self.residual_model.fit(interaction_features, residuals)
            
            # 模型诊断：检查残差模型的有效性
            residual_coefs = self.residual_model.coef_
            nonzero_coefs = np.sum(residual_coefs != 0)
            total_coefs = len(residual_coefs)
            nonzero_ratio = nonzero_coefs / total_coefs if total_coefs > 0 else 0
            
            print(f"残差模型诊断:")
            print(f"  - 总交互特征数: {total_coefs}")
            print(f"  - 非零系数数: {nonzero_coefs}")
            print(f"  - 非零系数比例: {nonzero_ratio:.3f}")
            
            if nonzero_coefs == 0:
                print("  ⚠️  警告：残差模型所有系数都为0，交互修正模块未起作用！")
            elif nonzero_ratio < 0.1:
                print("  ⚠️  警告：残差模型非零系数比例很低，交互修正效果可能有限")
            else:
                print("  ✅ 残差模型正常激活，交互修正模块工作正常")
            
            # 计算残差修正效果
            y_pred_residual = self.residual_model.predict(interaction_features)
            residual_pcc = np.corrcoef(residuals, y_pred_residual)[0, 1]
            print(f"残差修正模型PCC: {residual_pcc:.4f}")
        else:
            print("警告：无法创建交互特征，跳过残差修正阶段")
            self.residual_model = None
        
        self.is_trained = True
        print("GR²-LM模型训练完成！")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 主效应预测
        y_pred_main = self.main_model.predict(X_scaled)
        
        # 残差修正预测
        if self.residual_model is not None and self.snp_groups is not None:
            try:
                # 提取组代表特征
                group_features, _ = self._extract_group_representatives(X_scaled, self.snp_groups)
                
                # 创建交互特征
                interaction_features, _ = self._create_interaction_features(group_features)
                
                if interaction_features.size > 0:
                    y_pred_residual = self.residual_model.predict(interaction_features)
                    return y_pred_main + y_pred_residual
            except Exception as e:
                print(f"残差修正预测失败: {e}，仅使用主效应预测")
        
        return y_pred_main
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            Dict[str, float]: 评估指标，主要指标为PCC（皮尔逊相关系数）
        """
        y_pred = self.predict(X)
        
        # 计算皮尔逊相关系数（PCC）
        pcc = np.corrcoef(y, y_pred)[0, 1]
        
        metrics = {
            'pcc': pcc,  # 主要评估指标
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mae': np.mean(np.abs(y - y_pred))
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            Dict[str, float]: 特征重要性字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        importance_dict = {}
        
        # 主效应特征重要性
        if self.feature_names is None:
            feature_names = [f"SNP_{i}" for i in range(len(self.main_model.coef_))]
        else:
            feature_names = self.feature_names
        
        main_importance = np.abs(self.main_model.coef_)
        for name, importance in zip(feature_names, main_importance):
            importance_dict[f"Main_{name}"] = importance
        
        # 残差修正特征重要性
        if self.residual_model is not None:
            residual_importance = np.abs(self.residual_model.coef_)
            interaction_names = []
            
            # 重建交互特征名称
            n_groups = len(self.group_representatives)
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    interaction_names.append(f"Interaction_{i}_{j}")
            
            for name, importance in zip(interaction_names, residual_importance):
                importance_dict[f"Residual_{name}"] = importance
        
        return importance_dict
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        获取模型摘要信息
        
        Returns:
            Dict[str, Any]: 模型摘要
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        summary = {
            'model_type': 'GR²-LM',
            'task_type': self.task_type,
            'parameters': self.model_params,
            'n_snp_groups': len(self.snp_groups) if self.snp_groups else 0,
            'group_sizes': [len(group) for group in self.snp_groups] if self.snp_groups else [],
            'main_model_coef_nonzero': np.sum(self.main_model.coef_ != 0),
            'main_model_coef_total': len(self.main_model.coef_)
        }
        
        if self.residual_model is not None:
            summary['residual_model_coef_nonzero'] = np.sum(self.residual_model.coef_ != 0)
            summary['residual_model_coef_total'] = len(self.residual_model.coef_)
        
        return summary

    def get_params(self, deep=True):
        """
        支持sklearn参数搜索接口，返回所有可调参数
        """
        params = self.model_params.copy() if hasattr(self, 'model_params') else {}
        params['task_type'] = self.task_type
        return params

    def set_params(self, **params):
        """
        支持sklearn参数搜索接口，设置参数
        """
        # 初始化model_params如果不存在
        if not hasattr(self, 'model_params'):
            self.model_params = {}
        
        for key, value in params.items():
            if key == 'task_type':
                self.task_type = value
            else:
                # 将参数添加到model_params字典中
                self.model_params[key] = value
        
        # 重新初始化模型
        self._init_model()
        return self 