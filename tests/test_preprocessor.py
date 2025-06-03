"""
测试基因型数据预处理模块
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path
import pandas as pd
import logging

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.loader import DataLoader
from src.data.preprocessor import GenotypePreprocessor

logging.basicConfig(level=logging.INFO)

class TestGenotypePreprocessor(unittest.TestCase):
    """测试基因型数据预处理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.preprocessor = GenotypePreprocessor(
            maf_threshold=0.05,
            missing_threshold=0.1,
            gwas_p_threshold=1e-5,
            top_n_snps=3000
        )
        
        # 加载真实数据
        self.vcf_path = "data/Wheat1k.recode.vcf"
        self.pheno_path = "data/wheat1k.pheno.txt"
        
        # 检查文件是否存在
        self.assertTrue(Path(self.vcf_path).exists(), "VCF文件不存在")
        self.assertTrue(Path(self.pheno_path).exists(), "表型文件不存在")
    
    def test_real_data_preprocessing(self):
        """测试真实数据预处理"""
        # 1. 加载数据
        loader = DataLoader()
        genotype_matrix, snp_ids, sample_ids = loader.load_genotype(self.vcf_path)
        phenotype_data = pd.read_csv(self.pheno_path, sep='\t')
        
        # 2. 对齐样本
        # 从表型数据中提取HF编号
        phenotype_data['HF'] = phenotype_data['sample'].str.extract('HF(\d+)').astype(int)
        # 从基因型数据中提取HF编号
        genotype_hf = np.array([int(sid.split('_')[0][2:]) for sid in sample_ids])
        # 找到匹配的样本
        common_samples = np.intersect1d(phenotype_data['HF'], genotype_hf)
        # 获取对应的索引
        phenotype_indices = [phenotype_data[phenotype_data['HF'] == hf].index[0] for hf in common_samples]
        genotype_indices = [np.where(genotype_hf == hf)[0][0] for hf in common_samples]
        
        # 3. 预处理数据
        filtered_matrix, filtered_snp_ids, filtered_sample_ids = self.preprocessor.preprocess(
            genotype_matrix[:, genotype_indices],
            snp_ids,
            [sample_ids[i] for i in genotype_indices],
            phenotype_data.loc[phenotype_indices, 'yield'].values
        )
        
        # 4. 验证结果
        self.assertGreater(len(filtered_snp_ids), 0, "没有保留任何SNP")
        self.assertEqual(len(filtered_sample_ids), len(genotype_indices), "样本数量发生变化")
        self.assertEqual(filtered_matrix.shape, (len(filtered_snp_ids), len(filtered_sample_ids)), "矩阵维度不匹配")
        
        # 5. 检查缺失值
        self.assertEqual(np.sum(filtered_matrix == -1), 0, "存在缺失值")
        
        # 6. 检查基因型值范围
        unique_values = np.unique(filtered_matrix)
        self.assertTrue(all(x in [0, 1, 2] for x in unique_values), "基因型值不在[0,1,2]范围内")
        
        # 7. 检查保留的SNP数量
        self.assertLessEqual(len(filtered_snp_ids), 3000, "保留的SNP数量超过限制")
        
        # 8. 输出统计信息
        print("\n预处理结果统计:")
        print(f"原始SNP数量: {len(snp_ids)}")
        print(f"保留SNP数量: {len(filtered_snp_ids)}")
        print(f"样本数量: {len(filtered_sample_ids)}")
        print(f"基因型值分布: {dict(zip(*np.unique(filtered_matrix, return_counts=True)))}")

if __name__ == '__main__':
    unittest.main() 