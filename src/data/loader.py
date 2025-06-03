"""
数据加载模块
用于加载 VCF 格式的基因型数据和表型数据
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载器类"""
    
    def __init__(self):
        pass
        
    def load_phenotype(self, file_path):
        """
        加载表型数据
        
        Args:
            file_path: 表型数据文件路径
            
        Returns:
            pandas.DataFrame: 表型数据
        """
        # 读取表型数据
        pheno_data = pd.read_csv(file_path, sep='\t')
        
        # 验证必要的列是否存在
        required_columns = ['sample', 'spikelength', 'spikelet', 'lodging', 'kernelspikelet', 
                          'height', 'headingdate', 'gns', 'FHB', 'cold', 'yield', 'tkw', 
                          'tiller', 'sterilspike', 'FD', 'Mature']
        missing_columns = [col for col in required_columns if col not in pheno_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return pheno_data
        
    def load_genotype(self, file_path):
        """
        加载基因型数据
        
        Args:
            file_path: VCF文件路径
            
        Returns:
            tuple: (基因型矩阵, SNP ID列表, 样本ID列表, 染色体号列表)
        """
        # 读取VCF文件
        with open(file_path, 'r') as f:
            # 读取头部信息
            header = None
            for line in f:
                if line.startswith('#CHROM'):
                    header = line.strip().split('\t')
                    break
            
            if header is None:
                raise ValueError("VCF文件格式错误：缺少样本行")
            
            # 获取样本ID（从第10列开始）
            sample_ids = header[9:]
            logger.info(f"VCF文件中的样本数量: {len(sample_ids)}")
            
            # 分析样本ID
            self._analyze_sample_ids(sample_ids)
            
            # 读取基因型数据
            genotypes = []
            snp_ids = []
            chroms = []
            
            # 跳过所有注释行
            while True:
                line = f.readline()
                if not line.startswith('#'):
                    break
            
            # 处理第一行数据
            fields = line.strip().split('\t')
            chrom = fields[0]  # CHROM列
            snp_id = fields[2]  # ID列
            chroms.append(chrom)
            snp_ids.append(snp_id)
            
            # 获取基因型数据（从第10列开始）
            gt_data = []
            for gt in fields[9:]:
                if gt == '0/0':
                    gt_data.append(0)
                elif gt == '0/1':
                    gt_data.append(1)
                elif gt == '1/1':
                    gt_data.append(2)
                else:  # 处理缺失值
                    gt_data.append(-1)
            genotypes.append(gt_data)
            
            # 读取剩余的数据行
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 10:  # 跳过格式不正确的行
                    continue
                chrom = fields[0]
                snp_id = fields[2]  # ID列
                chroms.append(chrom)
                snp_ids.append(snp_id)
                
                # 获取基因型数据（从第10列开始）
                gt_data = []
                for gt in fields[9:]:
                    if gt == '0/0':
                        gt_data.append(0)
                    elif gt == '0/1':
                        gt_data.append(1)
                    elif gt == '1/1':
                        gt_data.append(2)
                    else:  # 处理缺失值
                        gt_data.append(-1)
                genotypes.append(gt_data)
        
        # 转换为numpy数组
        genotype_matrix = np.array(genotypes)
        
        # 验证样本数量
        if genotype_matrix.shape[1] != 998:
            logger.warning(f"样本数量 ({genotype_matrix.shape[1]}) 与预期值 (998) 不匹配")
            
        logger.info(f"加载的SNP数量: {len(snp_ids)}")
        logger.info(f"加载的样本数量: {len(sample_ids)}")
        
        return genotype_matrix, snp_ids, sample_ids, chroms

    def _analyze_sample_ids(self, sample_ids: List[str]):
        """
        分析样本ID，统计缺失的HF标识
        
        Args:
            sample_ids: 样本ID列表
        """
        # 提取所有HF编号
        hf_numbers = []
        for sample_id in sample_ids:
            match = re.match(r'HF(\d+)', sample_id)
            if match:
                hf_numbers.append(int(match.group(1)))
        
        # 找出缺失的编号
        if hf_numbers:
            min_hf = min(hf_numbers)
            max_hf = max(hf_numbers)
            all_hf = set(range(min_hf, max_hf + 1))
            missing_hf = all_hf - set(hf_numbers)
            
            logger.info(f"HF编号范围: {min_hf} - {max_hf}")
            logger.info(f"缺失的HF编号: {sorted(missing_hf)}")
            logger.info(f"缺失的HF编号数量: {len(missing_hf)}")

    @staticmethod
    def _convert_genotype(gt: Tuple) -> int:
        """
        将基因型转换为数值
        
        Args:
            gt: 基因型元组 (例如 (0, 0, True) 表示 0/0)
            
        Returns:
            int: 转换后的基因型值
                0: 参考基因型 (0/0)
                1: 杂合子 (0/1)
                2: 变异基因型 (1/1)
                -1: 缺失值
        """
        if gt is None or gt[0] is None or gt[1] is None:  # 缺失值
            return -1
        return gt[0] + gt[1]  # 0+0=0, 0+1=1, 1+1=2 