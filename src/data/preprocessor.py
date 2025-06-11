"""
基因型数据预处理器
用于处理基因型数据，包括缺失值填充、MAF过滤和GWAS分析
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List
from scipy import stats

logger = logging.getLogger(__name__)

class GenotypePreprocessor:
    """基因型数据预处理器类"""
    
    def __init__(self, maf_threshold: float = 0.05,
                 missing_threshold: float = 0.2,
                 gwas_p_threshold: float = 0.05,
                 top_n_snps: int = 1000):
        """
        初始化预处理器
        
        Args:
            maf_threshold: 最小等位基因频率阈值
            missing_threshold: 缺失值比例阈值
            gwas_p_threshold: GWAS p值阈值
            top_n_snps: 选择的SNP数量
        """
        self.maf_threshold = maf_threshold
        self.missing_threshold = missing_threshold
        self.gwas_p_threshold = gwas_p_threshold
        self.top_n_snps = top_n_snps
        self.gwas_results = None
        
    def preprocess(self, genotype_matrix: np.ndarray,
                  snp_ids: List[str],
                  sample_ids: List[str],
                  phenotype: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        预处理基因型数据
        
        Args:
            genotype_matrix: 基因型矩阵 (SNP数 x 样本数)
            snp_ids: SNP ID列表
            sample_ids: 样本ID列表
            phenotype: 表型值数组
            
        Returns:
            Tuple[np.ndarray, List[str], List[str]]: 
                - 处理后的基因型矩阵
                - 保留的SNP ID列表
                - 保留的样本ID列表
        """
        logger.info("开始预处理基因型数据...")
        logger.info(f"输入数据形状: {genotype_matrix.shape}")
        
        # 1. 过滤缺失值
        logger.info("过滤缺失值...")
        missing_filtered_matrix, missing_filtered_snps = self._filter_missing(
            genotype_matrix, snp_ids
        )
        logger.info(f"缺失值过滤后保留的SNP数量: {len(missing_filtered_snps)}")
        
        # 2. 过滤MAF
        logger.info("过滤MAF...")
        maf_filtered_matrix, maf_filtered_snps = self._filter_maf(
            missing_filtered_matrix, missing_filtered_snps
        )
        logger.info(f"MAF过滤后保留的SNP数量: {len(maf_filtered_snps)}")
        
        # 3. 填充剩余的缺失值
        logger.info("填充缺失值...")
        filled_matrix = self._fill_missing_values(maf_filtered_matrix)
        
        # 4. 过滤基因型完全相同的SNP
        logger.info("过滤基因型完全相同的SNP...")
        unique_matrix, unique_snps = self._filter_identical_genotypes(filled_matrix, maf_filtered_snps)
        logger.info(f"过滤后保留的SNP数量: {len(unique_snps)}")
        
        # 5. GWAS分析
        logger.info("进行GWAS分析...")
        gwas_results = self._gwas_analysis(unique_matrix, phenotype)
        
        # 6. 选择SNP
        logger.info("选择SNP...")
        selected_matrix, selected_snps = self._select_snps(
            unique_matrix, unique_snps, gwas_results
        )
        logger.info(f"最终选择的SNP数量: {len(selected_snps)}")
        
        return selected_matrix, selected_snps, sample_ids
        
    def _fill_missing_values(self, genotype_matrix: np.ndarray) -> np.ndarray:
        """
        填充缺失值
        
        Args:
            genotype_matrix: 基因型矩阵
            
        Returns:
            np.ndarray: 填充后的基因型矩阵
        """
        filled_matrix = genotype_matrix.copy()
        for i in range(genotype_matrix.shape[0]):
            # 获取非缺失值的索引
            non_missing = genotype_matrix[i] != -1
            if np.any(non_missing):
                # 计算非缺失值的众数
                values = genotype_matrix[i][non_missing]
                mode = stats.mode(values, keepdims=False)[0]
                # 用众数填充缺失值
                filled_matrix[i][~non_missing] = mode
        return filled_matrix
        
    def _filter_maf(self, genotype_matrix: np.ndarray,
                   snp_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        过滤MAF
        
        Args:
            genotype_matrix: 基因型矩阵
            snp_ids: SNP ID列表
            
        Returns:
            Tuple[np.ndarray, List[str]]: 
                - 过滤后的基因型矩阵
                - 保留的SNP ID列表
        """
        mafs = np.mean(genotype_matrix, axis=1) / 2
        maf_mask = (mafs >= self.maf_threshold) & (mafs <= 1 - self.maf_threshold)
        return genotype_matrix[maf_mask], [snp_ids[i] for i in np.where(maf_mask)[0]]
        
    def _filter_missing(self, genotype_matrix: np.ndarray,
                       snp_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        过滤缺失值
        
        Args:
            genotype_matrix: 基因型矩阵
            snp_ids: SNP ID列表
            
        Returns:
            Tuple[np.ndarray, List[str]]: 
                - 过滤后的基因型矩阵
                - 保留的SNP ID列表
        """
        missing_rates = np.mean(genotype_matrix == -1, axis=1)
        missing_mask = missing_rates <= self.missing_threshold
        return genotype_matrix[missing_mask], [snp_ids[i] for i in np.where(missing_mask)[0]]
        
    def _filter_identical_genotypes(self, genotype_matrix: np.ndarray,
                                  snp_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        过滤基因型完全相同的SNP
        
        Args:
            genotype_matrix: 基因型矩阵
            snp_ids: SNP ID列表
            
        Returns:
            Tuple[np.ndarray, List[str]]: 
                - 过滤后的基因型矩阵
                - 保留的SNP ID列表
        """
        # 计算每个SNP的标准差
        stds = np.std(genotype_matrix, axis=1)
        # 保留标准差大于0的SNP
        valid_mask = stds > 0
        return genotype_matrix[valid_mask], [snp_ids[i] for i in np.where(valid_mask)[0]]
        
    def _gwas_analysis(self, genotype_matrix: np.ndarray,
                      phenotype: np.ndarray) -> pd.DataFrame:
        """
        进行GWAS分析
        
        Args:
            genotype_matrix: 基因型矩阵
            phenotype: 表型值数组
            
        Returns:
            pd.DataFrame: GWAS结果
        """
        n_snps = genotype_matrix.shape[0]
        results = []
        
        for i in range(n_snps):
            # 获取当前SNP的基因型
            snp = genotype_matrix[i]
            
            # 进行线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(snp, phenotype)
            
            # 保存结果
            results.append({
                'snp_id': i,
                'p_value': p_value,
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'std_err': std_err
            })
            
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存结果
        self.gwas_results = results_df
        
        return results_df
        
    def _select_snps(self, genotype_matrix: np.ndarray,
                    snp_ids: List[str],
                    gwas_results: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        选择SNP
        
        Args:
            genotype_matrix: 基因型矩阵
            snp_ids: SNP ID列表
            gwas_results: GWAS结果
            
        Returns:
            Tuple[np.ndarray, List[str]]: 
                - 选择后的基因型矩阵
                - 选择的SNP ID列表
        """
        # 按p值排序
        sorted_indices = gwas_results['p_value'].argsort()
        
        # 选择前top_n_snps个SNP
        selected_indices = sorted_indices[:self.top_n_snps]
        
        return genotype_matrix[selected_indices], [snp_ids[i] for i in selected_indices] 