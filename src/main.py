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
from src.data.processed_data_loader import ProcessedDataLoader
from utils.training import train_and_evaluate
from utils.evaluation import plot_feature_importance, plot_prediction_vs_actual, evaluate_model
from utils.ensemble_weight_loader import load_trait_weights

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

def create_output_dirs(base_dir: str, config: Dict[str, Any], trait: str, timestamp: str) -> Dict[str, str]:
    """
    创建输出目录
    
    Args:
        base_dir: 基础目录
        config: 配置字典
        trait: 性状名称
        timestamp: 时间戳
        
    Returns:
        Dict[str, str]: 输出目录字典
    """
    # 创建带时间戳的评估目录
    evaluate_dir = os.path.join(base_dir, config['data']['output_dir'], f'evaluate_{timestamp}')
    trait_dir = os.path.join(evaluate_dir, trait)
    model_dir = os.path.join(trait_dir, 'models')
    plot_dir = os.path.join(trait_dir, 'plots')
    for dir_path in [evaluate_dir, trait_dir, model_dir, plot_dir]:
        os.makedirs(dir_path, exist_ok=True)
        
    return {
        'evaluate': evaluate_dir,
        'trait': trait_dir,
        'model': model_dir,
        'plot': plot_dir
    }

def convert_numpy_types(obj):
    """将NumPy数据类型转换为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main(args=None):
    """主函数"""
    # 解析命令行参数
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
    
    # 初始化数据加载器
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loader = ProcessedDataLoader(data_dir=os.path.join(project_root, 'preprocess_data', 'datasets'))
    
    # 获取可用的性状
    available_traits = loader.get_available_traits()
    if not available_traits:
        logger.error("没有找到预处理数据，请先运行预处理脚本 python src/data/preprocess_data.py")
        return
        
    logger.info("找到预处理数据，将使用预处理后的数据集...")
    # 确定要使用的模型和性状
    models_to_use = args.models or list(config['models'].keys())
    traits_to_use = args.traits or available_traits
    
    # 加载Ensemble权重表（如有），优先从config读取trait_weights
    ensemble_trait_weights = config['models'].get('Ensemble', {}).get('trait_weights', {})
    ensemble_weight_path = 'report/ensemble_weights_by_trait.csv'
    if os.path.exists(ensemble_weight_path):
        trait_weights_dict = load_trait_weights(ensemble_weight_path)
    else:
        trait_weights_dict = {}
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理每个性状
    all_results = {}
    for trait in traits_to_use:
        logger.info(f"\n===== 性状: {trait} =====")
        output_dirs = create_output_dirs(project_root, config, trait, timestamp)
        
        # 获取可用的SNP数量
        snp_options = loader.get_available_snp_counts(trait)
        if not snp_options:
            logger.warning(f"性状 {trait} 没有可用的预处理数据，跳过")
            continue
            
        logger.info(f"可用的SNP数量: {snp_options}")
        trait_results = {}
        
        # 处理每个SNP数量
        for n_snps in snp_options:
            logger.info(f"\n----- SNP数量: {n_snps} -----")
            
            # 加载预处理数据
            dataset = loader.load_dataset(trait, n_snps)
            X_train = dataset['X_train']
            X_test = dataset['X_test']
            y_train = dataset['y_train']
            y_test = dataset['y_test']
            snp_ids = dataset['snp_ids']
            
            # 从训练集中划分出验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.1,
                random_state=42
            )
            
            logger.info(f"训练集形状: {X_train.shape}")
            logger.info(f"验证集形状: {X_val.shape}")
            logger.info(f"测试集形状: {X_test.shape}")
            
            # 训练每个模型
            for model_name in models_to_use:
                if model_name not in config['models']:
                    logger.warning(f"跳过未知模型: {model_name}")
                    continue
                logger.info(f"\n{'='*50}")
                logger.info(f"训练模型: {model_name}")
                # 自动注入Ensemble权重，优先用config中的trait_weights
                model_config = config['models'][model_name].copy()
                if model_name.lower() == 'ensemble':
                    weights = None
                    if trait in ensemble_trait_weights:
                        weights = ensemble_trait_weights[trait]
                        logger.info(f"Ensemble模型自动注入权重（来自config trait_weights）: {weights}")
                    elif trait in trait_weights_dict:
                        weights = trait_weights_dict[trait]
                        logger.info(f"Ensemble模型自动注入权重（来自csv）: {weights}")
                    if weights is not None:
                        model_config['weights'] = weights
                # 创建模型
                model = ModelFactory.create_model(model_name, model_config)
                
                # 参数搜索
                param_grid = None
                if args.tune:
                    param_grid = config['models'][model_name].get('param_grid')
                    if param_grid is None:
                        logger.warning(f"模型 {model_name} 没有定义参数网格，跳过参数搜索")
                
                # 训练和评估模型
                results = train_and_evaluate(
                    model, X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test,
                    param_grid=param_grid,
                    n_jobs=config['training']['n_jobs'],
                    output_dir=output_dirs['model'],
                    prediction_dir=output_dirs['plot'],
                    prefix=f"{trait}_{model_name}_{n_snps}",
                    config=config
                )
                
                # 保存结果
                trait_results[f"{model_name}_{n_snps}"] = {
                    'test_metrics': results['test_metrics'],
                    'top_features': results['top_features'],
                    'param_results': results['param_results']
                }
                
                # 绘制特征重要性
                if hasattr(model, 'feature_importances_'):
                    plot_feature_importance(
                        model.feature_importances_,
                        snp_ids,
                        output_dir=output_dirs['plot']
                    )
        
        all_results[trait] = trait_results
        
        # 保存当前性状的结果
        results_file = os.path.join(output_dirs['trait'], f'results.txt')
        with open(results_file, 'w') as f:
            f.write(f"评估时间: {timestamp}\n")
            f.write(f"性状: {trait}\n")
            f.write("="*50 + "\n\n")
            
            for model_key, model_results in trait_results.items():
                f.write(f"模型: {model_key}\n")
                f.write("-"*30 + "\n")
                
                # 保存测试集指标
                f.write("测试集评估结果:\n")
                metrics = model_results['test_metrics']
                for metric_name, value in metrics.items():
                    if metric_name != 'y_pred':  # 跳过预测值
                        f.write(f"{metric_name}: {value:.4f}\n")
                
                # 保存特征重要性
                if model_results['top_features']:
                    f.write("\n特征重要性 (Top 10):\n")
                    for feature, importance in model_results['top_features'][:10]:
                        f.write(f"{feature}: {importance:.4f}\n")
                
                # 保存参数搜索结果
                if model_results['param_results']:
                    f.write("\n参数搜索结果:\n")
                    if isinstance(model_results['param_results'], list):
                        # 如果是列表，显示最佳参数
                        if model_results['param_results']:
                            best_result = model_results['param_results'][0]  # 第一个是最佳结果
                            if 'params' in best_result:
                                f.write("最佳参数:\n")
                                for param, value in best_result['params'].items():
                                    f.write(f"  {param}: {value}\n")
                                if 'val_score' in best_result:
                                    f.write(f"最佳验证分数: {best_result['val_score']:.4f}\n")
                    else:
                        # 如果是字典，直接遍历
                        for param, value in model_results['param_results'].items():
                            f.write(f"{param}: {value}\n")
                
                f.write("\n" + "="*50 + "\n\n")
                
        logger.info(f"性状 {trait} 的结果已保存到: {results_file}")

if __name__ == '__main__':
    main() 