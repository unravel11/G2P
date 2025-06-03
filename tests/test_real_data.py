"""
使用真实数据测试数据加载器
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.loader import DataLoader

class TestRealDataLoader(unittest.TestCase):
    """测试真实数据加载"""
    
    def setUp(self):
        """设置测试环境"""
        # 设置数据文件路径
        self.vcf_path = os.path.join('data', 'Wheat1k.recode.vcf')
        self.pheno_path = os.path.join('data', 'wheat1k.pheno.txt')
        
        # 验证文件是否存在
        self.assertTrue(os.path.exists(self.vcf_path), f"VCF文件不存在: {self.vcf_path}")
        self.assertTrue(os.path.exists(self.pheno_path), f"表型数据文件不存在: {self.pheno_path}")
        
        # 初始化数据加载器
        self.loader = DataLoader()
    
    def test_load_phenotype(self):
        """测试表型数据加载"""
        # 加载表型数据
        pheno_data = self.loader.load_phenotype(self.pheno_path)
        
        # 验证数据形状
        self.assertEqual(pheno_data.shape[0], 1000, "样本数量应该为1000")
        self.assertEqual(pheno_data.shape[1], 16, "列数应该为16（15个性状 + 1个样本ID）")
        
        # 验证列名
        expected_columns = ['sample', 'spikelength', 'spikelet', 'lodging', 'kernelspikelet', 
                          'height', 'headingdate', 'gns', 'FHB', 'cold', 'yield', 'tkw', 
                          'tiller', 'sterilspike', 'FD', 'Mature']
        self.assertTrue(all(col in pheno_data.columns for col in expected_columns),
                       "缺少必要的列")
        
        # 验证数据类型
        self.assertTrue(pd.api.types.is_numeric_dtype(pheno_data['spikelength']),
                       "spikelength应该是数值类型")
    
    def test_load_genotype(self):
        """测试基因型数据加载"""
        # 加载基因型数据
        genotype_matrix, snp_ids, sample_ids = self.loader.load_genotype(self.vcf_path)
        
        # 验证数据形状
        self.assertEqual(genotype_matrix.shape[0], 201740, "SNP数量应该为201740")
        self.assertEqual(genotype_matrix.shape[1], 998, "样本数量应该为998")
        
        # 验证SNP和样本ID数量
        self.assertEqual(len(snp_ids), genotype_matrix.shape[0], "SNP ID数量不匹配")
        self.assertEqual(len(sample_ids), genotype_matrix.shape[1], "样本ID数量不匹配")
        
        # 验证基因型值范围
        unique_values = np.unique(genotype_matrix)
        self.assertTrue(all(val in [-1, 0, 1, 2] for val in unique_values),
                       "基因型值应该为-1（缺失）, 0（参考）, 1（杂合）或2（变异）")
        
        # 验证缺失值比例
        missing_ratio = np.sum(genotype_matrix == -1) / genotype_matrix.size
        self.assertLess(missing_ratio, 0.1, "缺失值比例应该小于10%")

if __name__ == '__main__':
    unittest.main() 