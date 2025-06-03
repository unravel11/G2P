"""
测试GWAS可视化模块
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.loader import DataLoader
from src.data.preprocessor import GenotypePreprocessor
from src.visualization.gwas_plot import GWASPlotter

class TestGWASPlotter(unittest.TestCase):
    """测试GWAS可视化器"""
    
    def setUp(self):
        """设置测试环境"""
        self.vcf_path = "data/Wheat1k.recode.vcf"
        self.pheno_path = "data/wheat1k.pheno.txt"
        
        # 检查文件是否存在
        self.assertTrue(Path(self.vcf_path).exists(), "VCF文件不存在")
        self.assertTrue(Path(self.pheno_path).exists(), "表型文件不存在")
        
        # 创建输出目录
        self.output_dir = Path("output/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_gwas_visualization(self):
        """测试GWAS可视化"""
        # 1. 加载数据
        loader = DataLoader()
        genotype_matrix, snp_ids, sample_ids, chroms = loader.load_genotype(self.vcf_path)
        phenotype_data = pd.read_csv(self.pheno_path, sep='\t')
        
        # 2. 对齐样本
        phenotype_data['HF'] = phenotype_data['sample'].str.extract('HF(\d+)').astype(int)
        genotype_hf = np.array([int(sid.split('_')[0][2:]) for sid in sample_ids])
        common_samples = np.intersect1d(phenotype_data['HF'], genotype_hf)
        phenotype_indices = [phenotype_data[phenotype_data['HF'] == hf].index[0] for hf in common_samples]
        genotype_indices = [np.where(genotype_hf == hf)[0][0] for hf in common_samples]
        
        # 3. 预处理数据
        preprocessor = GenotypePreprocessor(
            maf_threshold=0.05,
            missing_threshold=0.1,
            gwas_p_threshold=1e-5,
            top_n_snps=3000
        )
        
        filtered_matrix, filtered_snp_ids, filtered_sample_ids = preprocessor.preprocess(
            genotype_matrix[:, genotype_indices],
            snp_ids,
            [sample_ids[i] for i in genotype_indices],
            phenotype_data.loc[phenotype_indices, 'yield'].values
        )
        
        # 4. 获取GWAS结果（健壮性处理）
        p_values = []
        for i in range(len(filtered_snp_ids)):
            snp = filtered_matrix[i]
            phenotype = phenotype_data.loc[phenotype_indices, 'yield'].values
            mask = ~np.isnan(snp) & ~np.isnan(phenotype)
            if np.sum(mask) > 1 and len(np.unique(snp[mask])) > 1:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(snp[mask], phenotype[mask])
                    p_values.append(p_value)
                except Exception:
                    p_values.append(1.0)
            else:
                p_values.append(1.0)
        
        # 5. 构建染色体信息字典
        snp_to_chrom = {snp: chrom for snp, chrom in zip(snp_ids, chroms)}
        filtered_chroms = [snp_to_chrom[snp] for snp in filtered_snp_ids]
        chrom_info = {snp: chrom for snp, chrom in zip(filtered_snp_ids, filtered_chroms)}
        
        # 6. 创建可视化
        plotter = GWASPlotter(filtered_snp_ids, np.array(p_values), chromosome_info=chrom_info)
        
        # 7. 绘制并保存图形
        plotter.plot_manhattan(
            output_file=self.output_dir / "manhattan_plot.png",
            title="Wheat Yield GWAS Manhattan Plot"
        )
        
        plotter.plot_qq(
            output_file=self.output_dir / "qq_plot.png",
            title="Wheat Yield GWAS QQ Plot"
        )
        
        # 8. 验证输出文件是否存在
        self.assertTrue((self.output_dir / "manhattan_plot.png").exists())
        self.assertTrue((self.output_dir / "qq_plot.png").exists())

if __name__ == '__main__':
    unittest.main() 