"""
CNN模型测试脚本
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
import logging
from datetime import datetime
import argparse
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.models.cnn_model import CNNModel
from src.data.loader import DataLoader
from src.data.preprocessor import GenotypePreprocessor
from src.utils.evaluation import plot_feature_importance, plot_prediction_vs_actual
from src.utils.training import train_and_evaluate

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_output_dirs(base_dir: str, trait: str) -> dict:
    """创建输出目录"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f'cnn_test_{current_time}')
    
    # 创建子目录
    trait_dir = os.path.join(output_dir, trait)
    importance_dir = os.path.join(trait_dir, 'feature_importance')
    prediction_dir = os.path.join(trait_dir, 'prediction_plots')
    
    # 创建所有目录
    for directory in [trait_dir, importance_dir, prediction_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return {
        'base': output_dir,
        'trait': trait_dir,
        'importance': importance_dir,
        'prediction': prediction_dir
    }

def test_cnn_model(trait: str = None, force_preprocess: bool = False):
    """测试CNN模型"""
    logger.info("开始测试CNN模型...")
    
    # 加载配置
    config = load_config(os.path.join(project_root, 'src', 'config.json'))
    cnn_config = config['models']['CNN']
    
    # 创建输出目录
    output_dirs = create_output_dirs(config['data']['output_dir'], trait or 'test_trait')
    
    # 直接进行数据加载和预处理
    logger.info("加载数据...")
    data_loader = DataLoader()
    logger.info("加载基因型数据...")
    geno_matrix, snp_ids, sample_ids, chroms = data_loader.load_genotype(config['data']['geno_file'])
    logger.info("加载表型数据...")
    pheno_data = data_loader.load_phenotype(config['data']['pheno_file'])
    pheno_data = pheno_data.set_index('sample')
    common_samples = list(set(sample_ids) & set(pheno_data.index))
    logger.info(f"共同样本数量: {len(common_samples)}")
    pheno_data = pheno_data.loc[common_samples]
    sample_indices = [sample_ids.index(sample) for sample in common_samples]
    geno_matrix = geno_matrix[:, sample_indices]
    sample_ids = [sample_ids[i] for i in sample_indices]
    logger.info(f"筛选后的表型数据形状: {pheno_data.shape}")
    logger.info(f"筛选后的基因型数据形状: {geno_matrix.shape}")
    
    # 数据预处理
    logger.info("数据预处理...")
    preprocessor = GenotypePreprocessor(
        maf_threshold=config['preprocessing']['maf_threshold'],
        missing_threshold=config['preprocessing']['missing_threshold'],
        gwas_p_threshold=config['preprocessing']['gwas_p_threshold'],
        top_n_snps=config['preprocessing']['top_n_snps']
    )
    
    if trait is None:
        trait = pheno_data.columns[1]
    logger.info(f"使用性状 {trait} 进行测试")
    
    # 预处理基因型数据
    filtered_matrix, filtered_snp_ids, filtered_sample_ids = preprocessor.preprocess(
        geno_matrix,
        snp_ids,
        sample_ids,
        phenotype=pheno_data[trait].values
    )
    
    # 准备特征和目标变量
    X = filtered_matrix.T.astype(np.float32)
    y = pheno_data.loc[filtered_sample_ids, trait].values.astype(np.float32)
    feature_names = filtered_snp_ids
    
    logger.info(f"数据形状: X={X.shape}, y={y.shape}")
    logger.info(f"数据类型: X={X.dtype}, y={y.dtype}")
    
    # 使用GWAS选择top SNPs
    if X.shape[1] > config['preprocessing']['top_n_snps']:
        logger.info(f"使用GWAS选择top {config['preprocessing']['top_n_snps']} SNPs")
        X, feature_names = preprocessor.select_top_snps(
            X, y, feature_names,
            n_snps=config['preprocessing']['top_n_snps']
        )
    
    # 划分训练集、验证集和测试集
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.2,  # 20%作为验证集
        random_state=config['training']['random_state']
    )
    
    # 训练和评估模型
    logger.info("开始训练和评估模型...")
    model = CNNModel(model_params=cnn_config, task_type='regression')
    results = train_and_evaluate(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dirs['trait'],
        prediction_dir=output_dirs['prediction'],
        prefix=f"cnn_{trait}"
    )
    logger.info("评估结果:")
    for metric_name, value in results['test_metrics'].items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # 获取特征重要性
    logger.info("获取特征重要性...")
    feature_importance = results['top_features']
    
    # 保存特征重要性
    importance_file = os.path.join(output_dirs['importance'], 'feature_importance.txt')
    with open(importance_file, 'w') as f:
        for feature, importance in feature_importance:
            f.write(f"{feature}: {importance:.4f}\n")
    
    # 绘制特征重要性图
    plot_feature_importance(
        list(feature_importance),
        top_n=10,
        output_dir=output_dirs['importance'],
        prefix=f"cnn_{trait}"
    )
    
    # 保存模型
    logger.info("保存模型...")
    model_path = os.path.join(output_dirs['trait'], 'model.pt')
    model.save(model_path)
    logger.info(f"模型已保存到: {model_path}")
    
    # 加载模型并验证
    logger.info("加载模型并验证...")
    new_model = CNNModel(model_params=cnn_config, task_type='regression')
    new_model.load(model_path)
    
    # 验证加载的模型
    new_metrics = new_model.evaluate(X_test, y_test)
    logger.info("加载模型的评估指标:")
    for metric_name, value in new_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    logger.info(f"测试完成! 结果保存在: {output_dirs['base']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试CNN模型')
    parser.add_argument('--trait', type=str, help='要预测的性状名称')
    parser.add_argument('--force-preprocess', action='store_true', help='强制重新进行数据预处理')
    args = parser.parse_args()
    test_cnn_model(args.trait, args.force_preprocess) 