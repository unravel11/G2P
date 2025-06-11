#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序入口
使用随机森林模型进行基因型-表型预测
"""

import numpy as np
import logging
import os
import json
import argparse
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib
from datetime import datetime
from sklearn.model_selection import train_test_split

from models.factory import ModelFactory
from data.loader import DataLoader
from data.preprocessor import GenotypePreprocessor
from utils.training import train_and_evaluate
from utils.evaluation import plot_feature_importance, plot_prediction_vs_actual, evaluate_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    保存评估结果
    
    Args:
        results: 评估结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建总体评估结果文件
    with open(os.path.join(output_dir, 'overall_evaluation.txt'), 'w') as f:
        f.write("模型评估结果汇总\n")
        f.write("=" * 50 + "\n\n")
        
        for trait, trait_res in results.items():
            f.write(f"性状: {trait}\n")
            f.write("=" * 40 + "\n\n")
            
            for model_name, res in trait_res.items():
                f.write(f"模型: {model_name}\n")
                
                # 获取模型参数
                model_params = res['model'].model_params
                f.write("模型参数:\n")
                for param_name, param_value in model_params.items():
                    f.write(f"- {param_name}: {param_value}\n")
                f.write("\n")
                
                # 如果有参数搜索结果，保存所有参数组合的结果
                if res.get('param_results'):
                    f.write("参数搜索结果:\n")
                    f.write("-" * 20 + "\n")
                    for i, result in enumerate(res['param_results'], 1):
                        f.write(f"组合 {i}:\n")
                        f.write("参数:\n")
                        for param_name, param_value in result['params'].items():
                            f.write(f"  - {param_name}: {param_value}\n")
                        f.write(f"测试集得分: {result['mean_test_score']:.4f} (±{result['std_test_score']:.4f})\n")
                        f.write(f"训练集得分: {result['mean_train_score']:.4f} (±{result['std_train_score']:.4f})\n")
                        f.write(f"排名: {result['rank_test_score']}\n\n")
                    f.write("\n")
                
                f.write("测试集评估结果:\n")
                for metric_name, value in res['test_metrics'].items():
                    f.write(f"- {metric_name}: {value:.4f}\n")
                f.write("\n")
                
                f.write("Top 10 重要SNP:\n")
                for snp_id, importance in res['top_features']:
                    f.write(f"- {snp_id}: {importance:.4f}\n")
                f.write("\n" + "=" * 50 + "\n\n")
    
    # 为每个性状创建单独的评估结果文件
    for trait, trait_res in results.items():
        trait_dir = os.path.join(output_dir, trait)
        os.makedirs(trait_dir, exist_ok=True)
        
        # 保存每个模型的评估结果
        for model_name, res in trait_res.items():
            model_dir = os.path.join(trait_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存模型参数
            with open(os.path.join(model_dir, 'model_params.txt'), 'w') as f:
                for param_name, param_value in res['model'].model_params.items():
                    f.write(f"{param_name}: {param_value}\n")
            
            # 保存测试集评估结果
            with open(os.path.join(model_dir, 'test_metrics.txt'), 'w') as f:
                for metric_name, value in res['test_metrics'].items():
                    f.write(f"{metric_name}: {value:.4f}\n")
            
            # 保存特征重要性
            with open(os.path.join(model_dir, 'feature_importance.txt'), 'w') as f:
                for snp_id, importance in res['top_features']:
                    f.write(f"{snp_id}: {importance:.4f}\n")

def create_output_dirs(project_root: str, config: Dict[str, Any], trait: str) -> Dict[str, str]:
    """
    创建输出目录结构
    
    Args:
        project_root: 项目根目录
        config: 配置字典
        trait: 当前处理的性状
        
    Returns:
        Dict[str, str]: 包含各个输出目录路径的字典
    """
    # 创建带日期标签的输出目录（只在第一次调用时创建）
    if not hasattr(create_output_dirs, 'base_output_dir'):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        create_output_dirs.base_output_dir = os.path.join(project_root, config['data']['output_dir'], f'evaluation_results_{current_time}')
        # 更新配置文件中的输出目录
        config['data']['output_dir'] = create_output_dirs.base_output_dir
    
    # 创建各个子目录
    trait_dir = os.path.join(create_output_dirs.base_output_dir, trait)
    importance_dir = os.path.join(trait_dir, 'feature_importance')
    prediction_dir = os.path.join(trait_dir, 'prediction_plots')
    loss_curve_dir = os.path.join(trait_dir, 'loss_curve')
    
    # 创建所有目录
    for directory in [trait_dir, importance_dir, prediction_dir, loss_curve_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return {
        'base': create_output_dirs.base_output_dir,
        'trait': trait_dir,
        'importance': importance_dir,
        'prediction': prediction_dir,
        'loss_curve': loss_curve_dir
    }

def main(args=None):
    # 设置随机种子以确保结果可重复
    np.random.seed(42)
    
    # 如果没有提供参数，则创建默认参数
    if args is None:
        parser = argparse.ArgumentParser(description='基因型-表型预测')
        parser.add_argument('--config', type=str, default='config.json',
                          help='配置文件路径')
        parser.add_argument('--models', type=str, nargs='+',
                          help='要使用的模型列表，例如：RandomForest XGBoost')
        parser.add_argument('--traits', type=str, nargs='+',
                          help='要预测的性状列表，例如：spikelength yield')
        parser.add_argument('--tune', action='store_true',
                          help='是否进行参数搜索')
        parser.add_argument('--n_jobs', type=int, default=-1,
                          help='并行计算的CPU核心数，-1表示使用所有可用核心，1表示不使用并行计算')
        args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 如果指定了n_jobs，覆盖配置文件中的设置
    if args.n_jobs is not None:
        config['training']['n_jobs'] = args.n_jobs
    
    # 1. 加载数据
    logger.info("正在加载数据...")
    loader = DataLoader()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pheno_file = os.path.join(project_root, config['data']['pheno_file'])
    geno_file = os.path.join(project_root, config['data']['geno_file'])
    pheno_data = loader.load_phenotype(pheno_file)
    genotype_matrix, snp_ids, sample_ids, chroms = loader.load_genotype(geno_file)
    pheno_data = pheno_data.set_index('sample')
    common_samples = list(set(sample_ids) & set(pheno_data.index))
    pheno_data = pheno_data.loc[common_samples]
    sample_indices = [sample_ids.index(sample) for sample in common_samples]
    genotype_matrix = genotype_matrix[:, sample_indices]
    sample_ids = [sample_ids[i] for i in sample_indices]
    logger.info(f"筛选后的表型数据形状: {pheno_data.shape}")
    logger.info(f"筛选后的基因型数据形状: {genotype_matrix.shape}")
    models_to_use = args.models or list(config['models'].keys())
    traits_to_use = args.traits or pheno_data.columns.tolist()
    preprocessor = GenotypePreprocessor(
        maf_threshold=config['preprocessing']['maf_threshold'],
        missing_threshold=config['preprocessing']['missing_threshold'],
        gwas_p_threshold=config['preprocessing']['gwas_p_threshold'],
        top_n_snps=config['preprocessing']['top_n_snps']
    )
    snp_options = config['preprocessing'].get('top_n_snps_grid', [config['preprocessing']['top_n_snps']])
    logger.info(f"SNP数量选项: {snp_options}")
    all_results = {}
    for trait in traits_to_use:
        logger.info(f"\n===== 性状: {trait} =====")
        output_dirs = create_output_dirs(project_root, config, trait)
        y_all = pheno_data[trait].values
        X_all = genotype_matrix.T  # shape: 样本数 x SNP数
        all_sample_ids = sample_ids  # 顺序对齐
        # 只划分一次训练集和测试集
        X_train, X_test, y_train, y_test, train_sample_ids, test_sample_ids = train_test_split(
            X_all, y_all, all_sample_ids, test_size=config['training']['test_size'], random_state=config['training']['random_state']
        )
        logger.info(f"训练集和测试集的大小: {X_train.shape}, {X_test.shape}")
        trait_results = {}
        for n_snps in snp_options:
            logger.info(f"\n----- SNP数量: {n_snps} -----")
            preprocessor.top_n_snps = n_snps
            # 只用训练集做GWAS，选SNP
            filtered_matrix_train, filtered_snp_ids, filtered_sample_ids = preprocessor.preprocess(
                X_train.T, snp_ids, train_sample_ids, y_train
            )
            X_train_selected = filtered_matrix_train.T
            snp_indices = [snp_ids.index(snp) for snp in filtered_snp_ids]
            X_test_selected = X_test[:, snp_indices]
            for model_name in models_to_use:
                if model_name not in config['models']:
                    logger.warning(f"跳过未知模型: {model_name}")
                    continue
                logger.info(f"\n{'='*50}")
                logger.info(f"训练模型: {model_name}")
                model = ModelFactory.create_model(model_name, config['models'][model_name])
                param_grid = None
                if args.tune:
                    param_grid = config['models'][model_name].get('param_grid')
                    if param_grid is None:
                        logger.warning(f"模型 {model_name} 没有定义参数网格，将使用默认参数")
                param_str = ""
                if args.tune and param_grid:
                    current_params = model.model_params
                    param_parts = []
                    for param_name, param_value in current_params.items():
                        if param_name in param_grid:
                            param_parts.append(f"{param_name}_{param_value}")
                    if param_parts:
                        param_str = "_" + "_".join(param_parts)
                snp_str = f"_snps_{n_snps}"
                # 划分训练集和验证集
                X_train_inner, X_val, y_train_inner, y_val = train_test_split(
                    X_train_selected, y_train, test_size=0.1, random_state=config['training']['random_state']
                )
                
                results = train_and_evaluate(
                    model=model,
                    X_train=X_train_inner,
                    y_train=y_train_inner,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test_selected,
                    y_test=y_test,
                    feature_names=filtered_snp_ids,
                    param_grid=param_grid,
                    n_jobs=config['training']['n_jobs'],
                    output_dir=output_dirs['loss_curve'],  # 用于保存loss曲线
                    prediction_dir=output_dirs['prediction'],  # 用于保存预测图
                    prefix=f"{model_name}{param_str}{snp_str}"
                )
                trait_results[f"{model_name}_{n_snps}"] = results
        all_results[trait] = trait_results
    save_results(all_results, output_dirs['base'])
    for trait, trait_res in all_results.items():
        for model_name, res in trait_res.items():
            param_str = ""
            if args.tune and res.get('param_results'):
                current_params = res['model'].model_params
                param_parts = []
                for param_name, param_value in current_params.items():
                    if param_name in config['models'][model_name.split('_')[0]].get('param_grid', {}):
                        param_parts.append(f"{param_name}_{param_value}")
                if param_parts:
                    param_str = "_" + "_".join(param_parts)
            n_snps = int(model_name.split('_')[-1])
            snp_str = f"_snps_{n_snps}"
            plot_feature_importance(
                res['top_features'],
                top_n=10,
                output_dir=output_dirs['importance'],
                prefix=f"{model_name}{param_str}{snp_str}"
            )

if __name__ == "__main__":
    main() 