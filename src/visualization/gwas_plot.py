"""
GWAS结果可视化模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)

class GWASPlotter:
    """GWAS结果可视化类"""
    
    def __init__(self, snp_ids, p_values, chromosome_info=None):
        """
        初始化GWAS可视化器
        
        参数:
            snp_ids: SNP ID列表
            p_values: 对应的p值列表
            chromosome_info: 染色体信息字典，格式为 {snp_id: chromosome}
        """
        self.snp_ids = np.array(snp_ids)
        self.p_values = np.array(p_values, dtype=float)  # 确保p值为浮点数
        self.chromosome_info = chromosome_info
        
        # 如果没有提供染色体信息，尝试从SNP ID中提取
        if chromosome_info is None:
            self.chromosome_info = self._extract_chromosome_info()
    
    def _extract_chromosome_info(self):
        """从SNP ID中提取染色体信息"""
        chromosome_info = {}
        for snp_id in self.snp_ids:
            # 假设SNP ID格式为 "chr1_12345" 或 "1_12345"
            try:
                chrom = snp_id.split('_')[0].replace('chr', '')
                chromosome_info[snp_id] = int(chrom)
            except:
                chromosome_info[snp_id] = 0
        return chromosome_info
    
    def plot_manhattan(self, output_file=None, title="GWAS Manhattan Plot", 
                      significance_threshold=1e-5, figsize=(12, 6)):
        """
        绘制Manhattan图
        
        参数:
            output_file: 输出文件路径，如果为None则显示图形
            title: 图形标题
            significance_threshold: 显著性阈值
            figsize: 图形大小
        """
        # 准备数据
        df = pd.DataFrame({
            'SNP': self.snp_ids,
            'P': self.p_values,
            'CHR': [self.chromosome_info[snp] for snp in self.snp_ids]
        })
        
        # 计算-log10(p)
        df['-log10(P)'] = -np.log10(df['P'].astype(float))  # 确保P列为浮点数
        
        # 按染色体排序
        df = df.sort_values(['CHR', 'SNP'])
        
        # 计算每个SNP的位置
        df['ind'] = range(len(df))
        
        # 设置图形
        plt.figure(figsize=figsize)
        
        # 为每个染色体使用不同颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 获取所有唯一染色体号，建立映射
        unique_chroms = list(df['CHR'].unique())
        chrom_to_idx = {chrom: idx for idx, chrom in enumerate(unique_chroms)}
        
        # 绘制每个染色体的点
        for chrom in unique_chroms:
            chrom_data = df[df['CHR'] == chrom]
            color = colors[chrom_to_idx[chrom] % len(colors)]
            plt.scatter(chrom_data['ind'], chrom_data['-log10(P)'],
                       color=color, alpha=0.6, label=f'Chr{chrom}')
        
        # 添加显著性阈值线
        plt.axhline(y=-np.log10(significance_threshold), color='r', linestyle='--',
                   label=f'Significance threshold ({significance_threshold})')
        
        # 设置图形属性
        plt.xlabel('SNP Position')
        plt.ylabel('-log10(P-value)')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存或显示图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_qq(self, output_file=None, title="GWAS QQ Plot", figsize=(8, 8)):
        """
        绘制QQ图
        
        参数:
            output_file: 输出文件路径，如果为None则显示图形
            title: 图形标题
            figsize: 图形大小
        """
        # 确保p值为浮点数
        p_values = self.p_values.astype(float)
        
        # 计算期望的p值
        expected = -np.log10(np.linspace(0, 1, len(p_values) + 1)[1:])
        observed = -np.log10(np.sort(p_values))
        
        # 计算lambda值（基因组膨胀因子）
        chi2 = stats.chi2.ppf(1 - p_values, 1)
        lambda_gc = np.median(chi2) / 0.455
        
        # 设置图形
        plt.figure(figsize=figsize)
        
        # 绘制散点图
        plt.scatter(expected, observed, alpha=0.6)
        
        # 添加对角线
        plt.plot([0, max(expected)], [0, max(expected)], 'r--')
        
        # 设置图形属性
        plt.xlabel('Expected -log10(P-value)')
        plt.ylabel('Observed -log10(P-value)')
        plt.title(f'{title}\nλ = {lambda_gc:.3f}')
        plt.tight_layout()
        
        # 保存或显示图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 