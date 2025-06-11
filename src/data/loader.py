"""
数据加载模块
用于加载原始基因型和表型数据
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载器类"""
    
    def load_phenotype(self, pheno_file: str) -> pd.DataFrame:
        """
        加载表型数据
        
        Args:
            pheno_file: 表型数据文件路径
            
        Returns:
            pd.DataFrame: 表型数据DataFrame
        """
        logger.info(f"加载表型数据: {pheno_file}")
        
        # 使用pandas直接读取文件
        df = pd.read_csv(pheno_file, sep='\t')
        
        # 将数值列转换为float类型
        numeric_cols = df.columns[1:]  # 除了sample列之外的所有列
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"表型数据形状: {df.shape}")
        return df
    
    def load_genotype(self, geno_file: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        加载基因型数据
        
        Args:
            geno_file: 基因型数据文件路径
            
        Returns:
            Tuple[np.ndarray, List[str], List[str]]: 
                - 基因型矩阵 (SNP数 x 样本数)
                - SNP ID列表
                - 样本ID列表
        """
        logger.info(f"加载基因型数据: {geno_file}")
        
        # 读取文件内容
        with open(geno_file, 'r') as f:
            lines = f.readlines()
        
        # 获取样本ID
        sample_ids = lines[0].strip().split()[9:]  # 跳过前9列
        
        # 初始化基因型矩阵和SNP ID列表
        genotype_matrix = []
        snp_ids = []
        
        # 处理每个SNP
        for line in lines[1:]:
            fields = line.strip().split()
            if len(fields) < 10:  # 跳过格式不正确的行
                continue
                
            # 获取SNP ID
            snp_id = fields[2] if fields[2] != '.' else f"{fields[0]}_{fields[1]}"
            snp_ids.append(snp_id)
            
            # 获取基因型
            genotypes = []
            for gt in fields[9:]:
                if gt == '0/0':
                    genotypes.append(0)
                elif gt == '0/1':
                    genotypes.append(1)
                elif gt == '1/1':
                    genotypes.append(2)
                else:  # './.' 或其他
                    genotypes.append(-1)
            
            genotype_matrix.append(genotypes)
        
        # 转换为numpy数组
        genotype_matrix = np.array(genotype_matrix)
        
        logger.info(f"基因型数据形状: {genotype_matrix.shape}")
        logger.info(f"SNP数量: {len(snp_ids)}")
        logger.info(f"样本数量: {len(sample_ids)}")
        
        return genotype_matrix, snp_ids, sample_ids 