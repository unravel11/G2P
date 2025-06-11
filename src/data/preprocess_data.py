"""
数据预处理模块
用于生成不同SNP数量的数据集
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.data.loader import DataLoader
from src.data.preprocessor import GenotypePreprocessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_dirs(base_dir: str) -> dict:
    """
    创建输出目录结构
    
    Args:
        base_dir: 基础输出目录
        
    Returns:
        dict: 包含各个子目录路径的字典
    """
    dirs = {
        'base': base_dir,
        'datasets': os.path.join(base_dir, 'datasets')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def save_dataset(X_train: np.ndarray, X_test: np.ndarray,
                y_train: np.ndarray, y_test: np.ndarray,
                snp_ids: list, output_dir: str, trait: str, snp_count: int):
    """
    保存数据集
    
    Args:
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
        snp_ids: SNP ID列表
        output_dir: 输出目录
        trait: 性状名称
        snp_count: SNP数量
    """
    # 创建性状目录
    trait_dir = os.path.join(output_dir, trait)
    os.makedirs(trait_dir, exist_ok=True)
    
    # 创建SNP数量目录
    snp_dir = os.path.join(trait_dir, f"{snp_count}_snps")
    os.makedirs(snp_dir, exist_ok=True)
    
    # 保存数据
    np.save(os.path.join(snp_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(snp_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(snp_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(snp_dir, 'y_test.npy'), y_test)
    
    # 保存SNP ID
    with open(os.path.join(snp_dir, 'snp_ids.txt'), 'w') as f:
        f.write('\n'.join(snp_ids))

def preprocess_data():
    """主预处理函数"""
    # 设置输入输出路径
    pheno_file = 'data/wheat1k.pheno.txt'
    geno_file = 'data/Wheat1k.recode.vcf'
    output_base = 'preprocess_data'
    
    # 创建输出目录
    output_dirs = create_output_dirs(output_base)
    
    # 加载原始数据
    logger.info("加载原始数据...")
    loader = DataLoader()
    pheno_data = loader.load_phenotype(pheno_file)
    geno_matrix, snp_ids, sample_ids = loader.load_genotype(geno_file)
    
    # 将表型数据的sample列设置为索引
    pheno_data.set_index('sample', inplace=True)
    
    # 确保样本对齐
    common_samples = list(set(pheno_data.index) & set(sample_ids))
    pheno_data = pheno_data.loc[common_samples]
    sample_indices = [sample_ids.index(sample) for sample in common_samples]
    geno_matrix = geno_matrix[:, sample_indices]
    
    logger.info(f"筛选后的表型数据形状: {pheno_data.shape}")
    logger.info(f"筛选后的基因型数据形状: {geno_matrix.shape}")
    
    # 定义SNP数量列表
    snp_counts = [100, 1000, 3000, 5000, 7000]
    
    # 对每个性状进行处理
    for trait in pheno_data.columns:
        logger.info(f"\n===== 性状: {trait} =====")
        
        # 获取当前性状的表型值
        y = pheno_data[trait].values
        
        # 创建预处理器
        preprocessor = GenotypePreprocessor(
            maf_threshold=0.05,
            missing_threshold=0.1,
            gwas_p_threshold=1e-5,
            top_n_snps=max(snp_counts)  # 使用最大的SNP数量
        )
        
        # 预处理基因型数据并进行GWAS分析
        logger.info("进行GWAS分析...")
        processed_geno, selected_snps, _ = preprocessor.preprocess(
            geno_matrix, snp_ids, common_samples, y
        )
        
        # 对每个SNP数量进行处理
        for snp_count in snp_counts:
            logger.info(f"\n处理 {snp_count} 个SNP...")
            
            # 选择指定数量的SNP
            current_geno = processed_geno[:snp_count]
            current_snps = selected_snps[:snp_count]
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                current_geno.T, y, test_size=0.2, random_state=42
            )
            
            # 保存数据集
            save_dataset(
                X_train, X_test, y_train, y_test,
                current_snps, output_dirs['datasets'],
                trait, snp_count
            )
            
            logger.info(f"已保存 {snp_count} 个SNP的数据集")

if __name__ == '__main__':
    preprocess_data() 