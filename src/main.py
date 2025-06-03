#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序入口
使用随机森林模型进行基因型-表型预测
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import RandomForestModel, XGBoostModel, LightGBMModel, LassoModel
from data.loader import DataLoader
from data.preprocessor import GenotypePreprocessor
from utils.evaluation import evaluate_model, cross_validate
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

matplotlib.rcParams['font.sans-serif'] = ['Arial']  # 或 ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def plot_feature_importance(feature_importance: dict, top_n: int = 10):
    """
    Plot feature importance bar chart
    Args:
        feature_importance: dict of feature importances
        top_n: number of top features to show
    """
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importance = zip(*top_features)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(importance), y=list(features))
    plt.title(f'Top {top_n} Important SNPs')
    plt.xlabel('Importance Score')
    plt.ylabel('SNP ID')
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plot predicted vs actual scatter plot
    Args:
        y_true: true values
        y_pred: predicted values
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Predicted vs Actual')
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
    plt.close()

def main():
    # 1. 加载数据
    logger.info("正在加载数据...")
    loader = DataLoader()
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 加载表型数据
    pheno_file = os.path.join(project_root, 'data', 'wheat1k.pheno.txt')
    logger.info(f"加载表型数据: {pheno_file}")
    pheno_data = loader.load_phenotype(pheno_file)
    logger.info(f"加载的表型数据形状: {pheno_data.shape}")
    
    # 加载基因型数据
    geno_file = os.path.join(project_root, 'data', 'Wheat1k.recode.vcf')
    logger.info(f"加载基因型数据: {geno_file}")
    genotype_matrix, snp_ids, sample_ids, chroms = loader.load_genotype(geno_file)
    logger.info(f"加载的基因型数据形状: {genotype_matrix.shape}")
    
    # 2. 数据预处理
    logger.info("正在预处理数据...")
    
    # 确保表型数据和基因型数据的样本匹配
    pheno_data = pheno_data.set_index('sample')
    common_samples = list(set(sample_ids) & set(pheno_data.index))
    logger.info(f"共同样本数量: {len(common_samples)}")
    
    # 按共同样本筛选数据
    pheno_data = pheno_data.loc[common_samples]
    sample_indices = [sample_ids.index(sample) for sample in common_samples]
    genotype_matrix = genotype_matrix[:, sample_indices]
    sample_ids = [sample_ids[i] for i in sample_indices]
    
    logger.info(f"筛选后的表型数据形状: {pheno_data.shape}")
    logger.info(f"筛选后的基因型数据形状: {genotype_matrix.shape}")
    
    preprocessor = GenotypePreprocessor(
        maf_threshold=0.05,
        missing_threshold=0.1,
        gwas_p_threshold=1e-5,
        top_n_snps=2000  # 选择top 2000个SNP
    )
    
    # 4. 定义所有模型及参数
    model_dict = {
        'RandomForest': (RandomForestModel, {
            'model_params': {
                'n_estimators': 500,         # 增加树的数量，提升稳定性
                'max_depth': 15,             # 适度加深树深度
                'min_samples_split': 4,      # 防止过拟合
                'min_samples_leaf': 2,       # 防止过拟合
                'max_features': 'sqrt',      # 每次分裂考虑部分特征
                'random_state': 42
            },
            'task_type': 'regression'
        }),
        'XGBoost': (XGBoostModel, {
            'model_params': {
                'n_estimators': 500,         # 增加树的数量
                'max_depth': 8,              # 适度加深
                'learning_rate': 0.05,       # 降低学习率，提升泛化
                'subsample': 0.8,            # 行采样，防止过拟合
                'colsample_bytree': 0.8,     # 列采样，防止过拟合
                'reg_alpha': 0.1,            # L1正则
                'reg_lambda': 1.0,           # L2正则
                'random_state': 42
            },
            'task_type': 'regression'
        }),
        'LightGBM': (LightGBMModel, {
            'model_params': {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbose': -1
            },
            'task_type': 'regression'
        }),
        'Lasso': (LassoModel, {
            'model_params': {
                'alpha': 0.01,               # 更小的alpha，避免特征全为0
                'max_iter': 10000,           # 增加迭代次数，保证收敛
                'random_state': 42
            }
        })
    }

    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 循环遍历所有性状
    trait_names = pheno_data.columns.tolist()
    all_results = {}
    for trait in trait_names:
        logger.info(f"===== Trait: {trait} =====")
        # 选择当前性状
        if trait != 'spikelength':
            continue
        target_phenotype = pheno_data[trait].values
        # 预处理基因型数据（GWAS筛选等）
        filtered_matrix, filtered_snp_ids, filtered_sample_ids = preprocessor.preprocess(
            genotype_matrix, snp_ids, sample_ids, target_phenotype
        )
        logger.info(f"Trait {trait}: filtered genotype shape: {filtered_matrix.shape}")
        # 确保样本顺序一致
        pheno_trait = pheno_data.loc[filtered_sample_ids]
        y = pheno_trait[trait].values
        scaler = StandardScaler()
        X = scaler.fit_transform(filtered_matrix.T)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        trait_results = {}
        for model_name, (ModelClass, kwargs) in model_dict.items():
            logger.info(f"Training {model_name} for trait {trait}...")
            model = ModelClass(**kwargs)
            model.train(X_train, y_train, feature_names=filtered_snp_ids)
            y_pred = model.predict(X_test)
            test_metrics = evaluate_model(y_test, y_pred)
            mean_metrics, fold_metrics = cross_validate(model, X, y, n_splits=5)
            feature_importance = model.get_feature_importance()
            # 保存特征重要性图
            plot_feature_importance(feature_importance, top_n=10)
            # 保存预测效果图
            plot_prediction_vs_actual(y_test, y_pred)
            trait_results[model_name] = {
                'test_metrics': test_metrics,
                'cv_metrics': mean_metrics,
                'fold_metrics': fold_metrics,
                'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            }
        all_results[trait] = trait_results
    # 统一保存所有性状、所有模型的评估结果
    with open(os.path.join(output_dir, 'model_evaluation.txt'), 'w') as f:
        f.write("模型评估结果\n")
        f.write("=" * 50 + "\n\n")
        for trait, trait_res in all_results.items():
            f.write(f"Trait: {trait}\n")
            f.write("-" * 40 + "\n")
            for model_name, res in trait_res.items():
                f.write(f"[{model_name}]\n")
                f.write("Test set metrics:\n")
                for metric_name, value in res['test_metrics'].items():
                    f.write(f"- {metric_name}: {value:.4f}\n")
                f.write("CV metrics:\n")
                for metric_name, value in res['cv_metrics'].items():
                    f.write(f"- {metric_name}: {value:.4f}\n")
                f.write("Top 10 Important SNPs:\n")
                for snp_id, importance in res['top_features']:
                    f.write(f"- {snp_id}: {importance:.4f}\n")
                f.write("\n")
            f.write("\n")

if __name__ == "__main__":
    main() 