"""
表型数据EDA分析测试脚本
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.analysis.phenotype_eda import PhenotypeEDA

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phenotype_eda(traits: list = None):
    """
    测试表型数据EDA分析
    
    Args:
        traits: 要分析的性状列表，如果为None则分析所有性状
    """
    logger.info("开始测试表型数据EDA分析...")
    
    # 从配置文件读取路径
    import json
    with open("src/config.json", "r") as f:
        config = json.load(f)
    
    # 初始化EDA分析器
    eda = PhenotypeEDA(output_dir=os.path.join(config["data"]["output_dir"], "eda"))
    
    # 运行分析
    results = eda.run_analysis(config["data"]["pheno_file"], traits)
    
    # 打印摘要
    print("\n===== 缺失和方差统计 =====")
    print(results["missing_variance"])
    
    print("\n===== 性状相关性矩阵 =====")
    print(results["correlation"])
    
    logger.info("EDA分析测试完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="表型数据EDA分析测试")
    parser.add_argument("--traits", nargs="+", help="要分析的性状列表，用空格分隔")
    args = parser.parse_args()
    
    test_phenotype_eda(args.traits)

if __name__ == "__main__":
    main() 