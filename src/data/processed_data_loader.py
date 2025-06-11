"""
预处理数据加载模块
用于加载预处理后的基因型-表型数据
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessedDataLoader:
    """预处理数据加载器类"""
    
    def __init__(self, data_dir: str = 'preprocess_data/datasets'):
        """
        初始化数据加载器
        
        Args:
            data_dir: 预处理数据目录
        """
        self.data_dir = data_dir
        
    def load_dataset(self, trait: str, n_snps: int) -> Dict:
        """
        加载指定性状和SNP数量的数据集
        
        Args:
            trait: 性状名称
            n_snps: SNP数量
            
        Returns:
            Dict: 包含训练集和测试集的字典
        """
        # 构建数据集路径
        dataset_dir = os.path.join(self.data_dir, trait, f"{n_snps}_snps")
        
        # 检查目录是否存在
        if not os.path.exists(dataset_dir):
            raise ValueError(f"数据集不存在: {dataset_dir}")
        
        # 加载数据
        X_train = np.load(os.path.join(dataset_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(dataset_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(dataset_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(dataset_dir, 'y_test.npy'))
        
        # 读取txt格式的SNP IDs
        with open(os.path.join(dataset_dir, 'snp_ids.txt'), 'r') as f:
            snp_ids = [line.strip() for line in f.readlines()]
        
        logger.info(f"加载数据集: {trait}, SNP数量: {n_snps}")
        logger.info(f"训练集形状: {X_train.shape}")
        logger.info(f"测试集形状: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'snp_ids': snp_ids,
            'sample_ids': None,  # 暂时设为None
            'info': {
                'trait': trait,
                'n_snps': n_snps,
                'n_samples_train': len(y_train),
                'n_samples_test': len(y_test)
            }
        }
    
    def get_available_traits(self) -> List[str]:
        """
        获取可用的性状列表
        
        Returns:
            List[str]: 性状名称列表
        """
        if not os.path.exists(self.data_dir):
            return []
        
        return [d for d in os.listdir(self.data_dir) 
                if os.path.isdir(os.path.join(self.data_dir, d))]
    
    def get_available_snp_counts(self, trait: str) -> List[int]:
        """
        获取指定性状可用的SNP数量列表
        
        Args:
            trait: 性状名称
            
        Returns:
            List[int]: SNP数量列表
        """
        trait_dir = os.path.join(self.data_dir, trait)
        if not os.path.exists(trait_dir):
            return []
        
        snp_dirs = [d for d in os.listdir(trait_dir) 
                   if d.endswith('_snps') and os.path.isdir(os.path.join(trait_dir, d))]
        
        return [int(d.split('_')[0]) for d in snp_dirs] 