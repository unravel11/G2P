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

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.models.cnn_model import CNNModel
from src.data.loader import DataLoader
from src.data.preprocessor import GenotypePreprocessor
from src.utils.evaluation import plot_feature_importance, plot_prediction_vs_actual

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

def test_cnn_model(trait: str = None):
    """测试CNN模型"""
    logger.info("开始测试CNN模型...")
    
    # 加载配置
    config = load_config(os.path.join(project_root, 'src', 'config.json'))
    cnn_config = config['models']['CNN']
    
    # 创建输出目录
    output_dirs = create_output_dirs(config['data']['output_dir'], trait or 'test_trait')
    
    # 加载数据
    logger.info("加载数据...")
    data_loader = DataLoader()
    
    # 加载基因型数据
    logger.info("加载基因型数据...")
    geno_matrix, snp_ids, sample_ids, chroms = data_loader.load_genotype(config['data']['geno_file'])
    
    # 加载表型数据
    logger.info("加载表型数据...")
    pheno_data = data_loader.load_phenotype(config['data']['pheno_file'])
    
    # 数据预处理
    logger.info("数据预处理...")
    preprocessor = GenotypePreprocessor(
        maf_threshold=config['preprocessing']['maf_threshold'],
        missing_threshold=config['preprocessing']['missing_threshold']
    )
    
    # 如果没有指定性状，使用第一个性状
    if trait is None:
        trait = pheno_data.columns[1]  # 跳过'sample'列
    logger.info(f"使用性状 {trait} 进行测试")
    
    # 预处理数据
    X, y, feature_names = preprocessor.preprocess(
        geno_matrix,
        pheno_data[trait]
    )
    
    # 如果SNP数量过多，使用GWAS选择top SNPs
    if X.shape[1] > config['preprocessing']['top_n_snps']:
        logger.info(f"使用GWAS选择top {config['preprocessing']['top_n_snps']} SNPs")
        X, feature_names = preprocessor.select_top_snps(
            X, y, feature_names,
            n_snps=config['preprocessing']['top_n_snps']
        )
    
    logger.info(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    
    # 初始化模型
    logger.info("初始化模型...")
    model = CNNModel(model_params=cnn_config, task_type='regression')
    
    # 训练模型
    logger.info("开始训练模型...")
    model.train(X_train, y_train, feature_names=feature_names)
    logger.info("模型训练完成")
    
    # 评估模型
    logger.info("评估模型性能...")
    metrics = model.evaluate(X_test, y_test)
    logger.info("评估指标:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # 获取特征重要性
    logger.info("获取特征重要性...")
    feature_importance = model.get_feature_importance()
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 保存特征重要性
    importance_file = os.path.join(output_dirs['importance'], 'feature_importance.txt')
    with open(importance_file, 'w') as f:
        for feature, importance in top_features:
            f.write(f"{feature}: {importance:.4f}\n")
    
    # 绘制特征重要性图
    plot_feature_importance(
        feature_importance,
        output_path=os.path.join(output_dirs['importance'], 'feature_importance.png')
    )
    
    # 绘制预测值与实际值对比图
    y_pred = model.predict(X_test)
    plot_prediction_vs_actual(
        y_test, y_pred,
        output_path=os.path.join(output_dirs['prediction'], 'prediction_vs_actual.png')
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
    args = parser.parse_args()
    test_cnn_model(args.trait) 