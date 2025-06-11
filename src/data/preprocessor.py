"""
基因型数据预处理模块
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from scipy import stats
import json
import os
import pandas as pd

logger = logging.getLogger(__name__)

class GenotypePreprocessor:
    """基因型数据预处理器"""
    
    def __init__(self, maf_threshold: float = 0.05, missing_threshold: float = 0.1,
                 gwas_p_threshold: float = 0.05, top_n_snps: Optional[int] = None):
        """
        初始化预处理器
        
        Args:
            maf_threshold: 最小等位基因频率阈值
            missing_threshold: 缺失值比例阈值
            gwas_p_threshold: GWAS显著性阈值
            top_n_snps: 选择的top N个SNP数量
        """
        self.maf_threshold = maf_threshold
        self.missing_threshold = missing_threshold
        self.gwas_p_threshold = gwas_p_threshold
        self.top_n_snps = top_n_snps
        
        # 设置随机种子
        np.random.seed(42)
        
        # 存储GWAS结果
        self.gwas_results = None
        self.gwas_metadata = None
        
    def preprocess(self, genotype_matrix: np.ndarray, snp_ids: List[str], 
                  sample_ids: List[str], phenotype: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        预处理基因型数据
        
        Args:
            genotype_matrix: 基因型矩阵
            snp_ids: SNP ID列表
            sample_ids: 样本ID列表
            phenotype: 表型数据，用于GWAS分析
            
        Returns:
            tuple: (处理后的基因型矩阵, 保留的SNP ID列表, 样本ID列表)
        """
        logger.info("开始预处理基因型数据...")
        
        # 处理空矩阵
        if genotype_matrix.size == 0:
            return np.array([]), [], sample_ids
        
        # 1. 基础过滤（MAF和缺失值）
        filtered_matrix, filtered_snp_ids, filtered_sample_ids = self._basic_filtering(
            genotype_matrix, snp_ids, sample_ids
        )
        
        # 2. 如果提供了表型数据，进行GWAS分析
        if phenotype is not None:
            filtered_matrix, filtered_snp_ids = self._gwas_selection(
                filtered_matrix, filtered_snp_ids, phenotype
            )
        
        logger.info(f"预处理完成:")
        logger.info(f"- 原始SNP数量: {len(snp_ids)}")
        logger.info(f"- 保留SNP数量: {len(filtered_snp_ids)}")
        logger.info(f"- 样本数量: {len(filtered_sample_ids)}")
        
        return filtered_matrix, filtered_snp_ids, filtered_sample_ids
    
    def _basic_filtering(self, genotype_matrix: np.ndarray, snp_ids: List[str], 
                        sample_ids: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
        """基础过滤（MAF和缺失值）"""
        # 1. 计算每个SNP的缺失值比例
        missing_ratio = np.sum(genotype_matrix == -1, axis=1) / genotype_matrix.shape[1]
        valid_snps = missing_ratio <= self.missing_threshold
        
        # 2. 计算每个SNP的MAF
        maf = self._calculate_maf(genotype_matrix)
        valid_snps_maf = maf >= self.maf_threshold
        
        # 3. 更新有效的SNP索引
        valid_snps = np.where(np.logical_and(valid_snps, valid_snps_maf))[0]
        
        # 4. 过滤SNP
        filtered_matrix = genotype_matrix[valid_snps]
        filtered_snp_ids = [snp_ids[i] for i in valid_snps]
        
        # 5. 填充缺失值
        filled_matrix = self._fill_missing_values(filtered_matrix)
        
        return filled_matrix, filtered_snp_ids, sample_ids
    
    def _gwas_selection(self, genotype_matrix: np.ndarray, snp_ids: List[str], 
                       phenotype: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """基于GWAS的SNP选择"""
        logger.info("开始GWAS分析...")
        
        # 1. 对每个SNP进行单变量回归
        p_values = []
        slopes = []
        intercepts = []
        r_values = []
        std_errs = []
        
        for i in range(genotype_matrix.shape[0]):
            # 获取当前SNP的基因型
            snp = genotype_matrix[i]
            # 进行线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(snp, phenotype)
            p_values.append(p_value)
            slopes.append(slope)
            intercepts.append(intercept)
            r_values.append(r_value)
            std_errs.append(std_err)
        
        # 2. 根据p值选择SNP
        p_values = np.array(p_values)
        significant_snps = p_values <= self.gwas_p_threshold
        
        # 3. 如果指定了top N，选择p值最小的N个SNP
        if self.top_n_snps is not None:
            top_indices = np.argsort(p_values)[:self.top_n_snps]
            significant_snps = np.zeros_like(significant_snps, dtype=bool)
            significant_snps[top_indices] = True
        
        # 4. 过滤SNP
        filtered_matrix = genotype_matrix[significant_snps]
        filtered_snp_ids = [snp_ids[i] for i in np.where(significant_snps)[0]]
        
        # 5. 保存GWAS结果
        self.gwas_results = {
            'snp_ids': snp_ids,
            'p_values': p_values.tolist(),
            'slopes': slopes,
            'intercepts': intercepts,
            'r_values': r_values,
        }
        
        self.gwas_metadata = {
            'maf_threshold': self.maf_threshold,
            'missing_threshold': self.missing_threshold,
            'gwas_p_threshold': self.gwas_p_threshold,
            'top_n_snps': self.top_n_snps,
            'significant_snps': significant_snps.tolist(),
            'filtered_snp_ids': filtered_snp_ids
        }
        
        logger.info(f"GWAS分析完成:")
        logger.info(f"- 显著SNP数量: {len(filtered_snp_ids)}")
        logger.info(f"- 最小p值: {min(p_values):.2e}")
        logger.info(f"- 最大p值: {max(p_values):.2e}")
        
        return filtered_matrix, filtered_snp_ids
    
    def _calculate_maf(self, genotype_matrix: np.ndarray) -> np.ndarray:
        """计算最小等位基因频率"""
        maf_values = []
        for i in range(genotype_matrix.shape[0]):
            # 获取非缺失值
            non_missing = genotype_matrix[i][genotype_matrix[i] != -1]
            if len(non_missing) > 0:
                # 计算等位基因频率（使用稳定的计算方式）
                allele_freq = np.sum(non_missing) / (2 * len(non_missing))
                # 计算MAF（确保使用相同的计算方式）
                maf = min(allele_freq, 1 - allele_freq)
            else:
                maf = 0
            maf_values.append(maf)
        
        return np.array(maf_values, dtype=np.float64)  # 确保使用相同的精度
    
    def _fill_missing_values(self, genotype_matrix: np.ndarray) -> np.ndarray:
        """使用众数填充缺失值"""
        filled_matrix = genotype_matrix.copy()
        
        for i in range(genotype_matrix.shape[0]):
            # 获取非缺失值
            non_missing = genotype_matrix[i][genotype_matrix[i] != -1]
            if len(non_missing) > 0:
                # 计算众数（使用稳定的排序）
                values, counts = np.unique(non_missing, return_counts=True)
                # 当有多个众数时，选择最小的那个以保持确定性
                max_count = np.max(counts)
                modes = values[counts == max_count]
                mode = np.min(modes)  # 选择最小的众数
                # 填充缺失值
                filled_matrix[i][genotype_matrix[i] == -1] = mode
        
        return filled_matrix 

    def save_gwas_results(self, output_dir: str, trait_name: str):
        """
        保存GWAS分析结果
        
        Args:
            output_dir: 输出目录
            trait_name: 性状名称
        """
        if self.gwas_results is None or self.gwas_metadata is None:
            raise ValueError("没有可用的GWAS结果")
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存GWAS结果
        results_df = pd.DataFrame({
            'snp_id': self.gwas_results['snp_ids'],
            'p_value': self.gwas_results['p_values'],
            'slope': self.gwas_results['slopes'],
            'intercept': self.gwas_results['intercepts'],
            'r_value': self.gwas_results['r_values']
        })
        
        # 保存为CSV文件
        results_file = os.path.join(output_dir, f'gwas_results_{trait_name}.csv')
        results_df.to_csv(results_file, index=False)
        
        # 保存元数据
        metadata_file = os.path.join(output_dir, f'gwas_metadata_{trait_name}.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.gwas_metadata, f, indent=4)
            
        logger.info(f"GWAS结果已保存到: {results_file}")
        logger.info(f"GWAS元数据已保存到: {metadata_file}")

    def load_gwas_results(self, results_file: str, metadata_file: str):
        """
        加载GWAS分析结果
        
        Args:
            results_file: GWAS结果文件路径
            metadata_file: GWAS元数据文件路径
        """
        # 加载GWAS结果
        results_df = pd.read_csv(results_file)
        self.gwas_results = {
            'snp_ids': results_df['snp_id'].tolist(),
            'p_values': results_df['p_value'].tolist(),
            'slopes': results_df['slope'].tolist(),
            'intercepts': results_df['intercept'].tolist(),
            'r_values': results_df['r_value'].tolist()
        }
        
        # 加载元数据
        with open(metadata_file, 'r') as f:
            self.gwas_metadata = json.load(f)
            
        # 更新预处理器参数
        self.maf_threshold = self.gwas_metadata['maf_threshold']
        self.missing_threshold = self.gwas_metadata['missing_threshold']
        self.gwas_p_threshold = self.gwas_metadata['gwas_p_threshold']
        self.top_n_snps = self.gwas_metadata['top_n_snps']
        
        logger.info("GWAS结果加载完成")
        logger.info(f"- 加载的SNP数量: {len(self.gwas_results['snp_ids'])}")
        logger.info(f"- 显著SNP数量: {sum(self.gwas_metadata['significant_snps'])}")

    def get_significant_snps(self) -> List[str]:
        """
        获取显著SNP列表
        
        Returns:
            List[str]: 显著SNP ID列表
        """
        if self.gwas_metadata is None:
            raise ValueError("没有可用的GWAS结果")
        return self.gwas_metadata['filtered_snp_ids']

    def get_gwas_summary(self) -> Dict:
        """
        获取GWAS分析摘要
        
        Returns:
            Dict: GWAS分析摘要信息
        """
        if self.gwas_results is None:
            raise ValueError("没有可用的GWAS结果")
            
        p_values = np.array(self.gwas_results['p_values'])
        return {
            'total_snps': len(self.gwas_results['snp_ids']),
            'significant_snps': sum(self.gwas_metadata['significant_snps']),
            'min_p_value': float(np.min(p_values)),
            'max_p_value': float(np.max(p_values)),
            'mean_p_value': float(np.mean(p_values)),
            'median_p_value': float(np.median(p_values))
        } 