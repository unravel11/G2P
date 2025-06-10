"""
表型数据EDA分析模块
用于分析小麦表型数据的分布特征和相关性
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, List, Dict
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhenotypeEDA:
    """表型数据EDA分析类"""
    
    def __init__(self, output_dir: str = "output/eda"):
        """
        初始化EDA分析器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self, pheno_file: str) -> pd.DataFrame:
        """
        加载表型数据
        
        Args:
            pheno_file: 表型数据文件路径
            
        Returns:
            pd.DataFrame: 表型数据DataFrame
        """
        logger.info(f"加载表型数据: {pheno_file}")
        return pd.read_csv(pheno_file, sep="\t")
    
    def analyze_missing_and_variance(self, pheno_df: pd.DataFrame) -> pd.DataFrame:
        """
        分析缺失值和方差
        
        Args:
            pheno_df: 表型数据DataFrame
            
        Returns:
            pd.DataFrame: 分析结果
        """
        logger.info("分析缺失值和方差...")
        missing = pheno_df.isnull().sum()
        variances = pheno_df.var(numeric_only=True).sort_values(ascending=False)
        
        summary_df = pd.DataFrame({
            "缺失值数量": missing,
            "缺失比例": missing / len(pheno_df),
            "方差": variances
        }).dropna()
        
        # 保存结果
        output_file = os.path.join(self.output_dir, "missing_variance_summary.csv")
        summary_df.to_csv(output_file)
        logger.info(f"分析结果已保存至: {output_file}")
        
        return summary_df
    
    def plot_distributions(self, pheno_df: pd.DataFrame, traits: Optional[List[str]] = None) -> None:
        """
        绘制性状分布图
        
        Args:
            pheno_df: 表型数据DataFrame
            traits: 要分析的性状列表，如果为None则分析所有性状
        """
        logger.info("绘制性状分布图...")
        if traits is None:
            traits = pheno_df.columns[1:]  # 去除第一列sample ID
            
        for trait in traits:
            plt.figure(figsize=(6, 4))
            sns.histplot(pheno_df[trait].dropna(), kde=True, bins=30)
            plt.title(f"Distribution of {trait}")
            plt.xlabel(trait)
            plt.ylabel("Count")
            plt.tight_layout()
            
            # 保存图片
            output_file = os.path.join(self.output_dir, f"{trait}_distribution.png")
            plt.savefig(output_file)
            plt.close()
            logger.info(f"分布图已保存至: {output_file}")
    
    def analyze_correlations(self, pheno_df: pd.DataFrame) -> pd.DataFrame:
        """
        分析性状间相关性
        
        Args:
            pheno_df: 表型数据DataFrame
            
        Returns:
            pd.DataFrame: 相关性矩阵
        """
        logger.info("分析性状间相关性...")
        corr_matrix = pheno_df.iloc[:, 1:].corr()
        
        # 绘制相关性热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Trait Correlation Matrix")
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(self.output_dir, "trait_correlation.png")
        plt.savefig(output_file)
        plt.close()
        logger.info(f"相关性热图已保存至: {output_file}")
        
        # 保存相关性矩阵
        corr_file = os.path.join(self.output_dir, "trait_correlation.csv")
        corr_matrix.to_csv(corr_file)
        logger.info(f"相关性矩阵已保存至: {corr_file}")
        
        return corr_matrix
    
    def run_analysis(self, pheno_file: str, traits: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        运行完整的EDA分析
        
        Args:
            pheno_file: 表型数据文件路径
            traits: 要分析的性状列表
            
        Returns:
            Dict[str, pd.DataFrame]: 分析结果字典
        """
        logger.info("开始表型数据EDA分析...")
        
        # 加载数据
        pheno_df = self.load_data(pheno_file)
        
        # 分析缺失值和方差
        missing_var_summary = self.analyze_missing_and_variance(pheno_df)
        
        # 绘制分布图
        self.plot_distributions(pheno_df, traits)
        
        # 分析相关性
        corr_matrix = self.analyze_correlations(pheno_df)
        
        logger.info("EDA分析完成")
        
        return {
            "missing_variance": missing_var_summary,
            "correlation": corr_matrix
        }

def main():
    """主函数"""
    # 从配置文件读取路径
    import json
    with open("src/config.json", "r") as f:
        config = json.load(f)
    
    # 初始化EDA分析器
    eda = PhenotypeEDA(output_dir=os.path.join(config["data"]["output_dir"], "eda"))
    
    # 运行分析
    results = eda.run_analysis(config["data"]["pheno_file"])
    
    # 打印摘要
    print("\n===== 缺失和方差统计 =====")
    print(results["missing_variance"])
    
    print("\n===== 性状相关性矩阵 =====")
    print(results["correlation"])

if __name__ == "__main__":
    main() 