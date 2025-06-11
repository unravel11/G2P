"""
数据预处理脚本：生成不同 SNP 数量的数据集
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.loader import DataLoader
from src.data.preprocessor import GenotypePreprocessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_dirs(base_dir: str, trait: str) -> dict:
    """创建输出目录"""
    output_dirs = {
        'base': os.path.join(base_dir, trait),
        'gwas': os.path.join(base_dir, trait, 'gwas'),
        'data': os.path.join(base_dir, trait, 'data')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return output_dirs

def save_dataset(X_train: np.ndarray, X_test: np.ndarray, 
                y_train: np.ndarray, y_test: np.ndarray,
                snp_ids: list, sample_ids: list,
                output_dir: str, n_snps: int):
    """保存数据集"""
    # 创建数据集目录
    dataset_dir = os.path.join(output_dir, f'snps_{n_snps}')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 保存训练集
    np.save(os.path.join(dataset_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(dataset_dir, 'y_train.npy'), y_train)
    
    # 保存测试集
    np.save(os.path.join(dataset_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(dataset_dir, 'y_test.npy'), y_test)
    
    # 保存 SNP ID 和样本 ID
    with open(os.path.join(dataset_dir, 'snp_ids.json'), 'w') as f:
        json.dump(snp_ids, f)
    
    with open(os.path.join(dataset_dir, 'sample_ids.json'), 'w') as f:
        json.dump(sample_ids, f)
    
    # 保存数据集信息
    info = {
        'n_snps': n_snps,
        'n_samples_train': len(y_train),
        'n_samples_test': len(y_test),
        'feature_names': snp_ids
    }
    
    with open(os.path.join(dataset_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    
    logger.info(f"数据集已保存到: {dataset_dir}")

def preprocess_data(config_path: str = 'src/config.json'):
    """预处理数据并生成不同 SNP 数量的数据集"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建数据加载器
    loader = DataLoader()
    
    # 加载基因型数据
    logger.info("加载基因型数据...")
    genotype_matrix, snp_ids, sample_ids, chroms = loader.load_genotype(config['data']['geno_file'])
    
    # 加载表型数据
    logger.info("加载表型数据...")
    pheno_data = pd.read_csv(config['data']['pheno_file'], sep='\t')
    pheno_data = pheno_data.set_index('sample')
    
    # 对齐样本
    common_samples = list(set(sample_ids) & set(pheno_data.index))
    logger.info(f"共同样本数量: {len(common_samples)}")
    
    # 获取样本索引
    sample_indices = [sample_ids.index(sample) for sample in common_samples]
    genotype_matrix = genotype_matrix[:, sample_indices]
    sample_ids = [sample_ids[i] for i in sample_indices]
    pheno_data = pheno_data.loc[common_samples]
    
    # 创建预处理器
    preprocessor = GenotypePreprocessor(
        maf_threshold=config['preprocessing']['maf_threshold'],
        missing_threshold=config['preprocessing']['missing_threshold'],
        gwas_p_threshold=config['preprocessing']['gwas_p_threshold']
    )
    
    # 对每个性状进行处理
    traits = pheno_data.columns
    snp_options = [1000, 3000, 5000, 8000]
    
    for trait in traits:
        logger.info(f"\n处理性状: {trait}")
        
        # 创建输出目录
        output_dirs = create_output_dirs(config['data']['output_dir'], trait)
        
        # 获取表型数据
        y = pheno_data[trait].values
        
        # 划分训练集和测试集
        X_all = genotype_matrix.T  # shape: 样本数 x SNP数
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y,
            test_size=0.2,
            random_state=config['training']['random_state']
        )
        
        # 对每个 SNP 数量进行处理
        for n_snps in snp_options:
            logger.info(f"\n处理 SNP 数量: {n_snps}")
            
            # 设置 SNP 数量
            preprocessor.top_n_snps = n_snps
            
            # 预处理训练集
            filtered_matrix_train, filtered_snp_ids, filtered_sample_ids = preprocessor.preprocess(
                X_train.T, snp_ids, sample_ids, y_train
            )
            
            # 保存 GWAS 结果
            preprocessor.save_gwas_results(
                output_dir=output_dirs['gwas'],
                trait_name=f"{trait}_snps_{n_snps}"
            )
            
            # 准备测试集数据
            snp_indices = [snp_ids.index(snp) for snp in filtered_snp_ids]
            X_test_selected = X_test[:, snp_indices]
            
            # 保存数据集
            save_dataset(
                X_train=filtered_matrix_train.T,
                X_test=X_test_selected,
                y_train=y_train,
                y_test=y_test,
                snp_ids=filtered_snp_ids,
                sample_ids=filtered_sample_ids,
                output_dir=output_dirs['data'],
                n_snps=n_snps
            )

if __name__ == '__main__':
    preprocess_data() 