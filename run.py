#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行脚本
用于从项目根目录运行G2P程序
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, List

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入并运行主程序
from src.main import main

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def print_tuning_tasks(config: Dict[str, Any], models: List[str], traits: List[str]) -> None:
    """打印参数搜索任务信息"""
    print("\n=== 参数搜索任务信息 ===")
    print(f"模型数量: {len(models)}")
    print(f"性状数量: {len(traits)}")
    print(f"SNP数量选项: {len(config['preprocessing']['top_n_snps_grid'])}")
    
    # 计算每个模型的参数组合数
    model_combinations = {}
    total_model_combinations = 0
    for model_name in models:
        if model_name in config['models']:
            param_grid = config['models'][model_name].get('param_grid', {})
            param_combinations = 1
            for param_name, param_values in param_grid.items():
                param_combinations *= len(param_values)
            model_combinations[model_name] = param_combinations
            total_model_combinations += param_combinations
    
    # 计算总任务数
    snp_options = len(config['preprocessing']['top_n_snps_grid'])
    total_tasks = total_model_combinations * snp_options * len(traits)
    
    print("\n参数组合详情:")
    print(f"- 每个性状的SNP数量选项: {snp_options}")
    print("- 每个模型的参数组合数:")
    for model_name, combinations in model_combinations.items():
        print(f"  * {model_name}: {combinations} 个组合")
    print(f"- 每个性状的模型参数组合总数: {total_model_combinations}")
    print(f"- 每个性状的总任务数: {total_model_combinations * snp_options}")
    print(f"\n总任务数: {total_tasks}")
    
    print("\n详细任务列表:")
    for trait in traits:
        print(f"\n性状: {trait}")
        print(f"SNP数量选项: {config['preprocessing']['top_n_snps_grid']}")
        for model_name in models:
            if model_name in config['models']:
                param_grid = config['models'][model_name].get('param_grid', {})
                param_combinations = 1
                for param_name, param_values in param_grid.items():
                    param_combinations *= len(param_values)
                print(f"  - {model_name}: {param_combinations} 个参数组合")
    print("\n" + "=" * 30 + "\n")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='基因型-表型预测')
    parser.add_argument('--config', type=str, default='src/config.json',
                      help='配置文件路径')
    parser.add_argument('--models', type=str, nargs='+',
                      help='要使用的模型列表，例如：RandomForest XGBoost')
    parser.add_argument('--traits', type=str, nargs='+',
                      help='要预测的性状列表，例如：spikelength yield')
    parser.add_argument('--tune', action='store_true',
                      help='是否进行参数搜索')
    parser.add_argument('--n_jobs', type=int, default=-1,
                      help='并行计算的CPU核心数，-1表示使用所有可用核心，1表示不使用并行计算')
    args = parser.parse_args()
    
    # 如果启用参数搜索，打印任务信息
    if args.tune:
        config = load_config(args.config)
        models_to_use = args.models or list(config['models'].keys())
        # 如果未指定性状，使用所有15个性状
        traits_to_use = args.traits or [
            'spikelength', 'spikeweight', 'spikeletnumber', 'spikeletdensity',
            'spikeletfertility', 'grainweight', 'grainlength', 'grainwidth',
            'grainthickness', 'grainarea', 'grainperimeter', 'graincircularity',
            'grainaspectratio', 'graincompactness', 'yield'
        ]
        print_tuning_tasks(config, models_to_use, traits_to_use)
    
    # 运行主程序
    main(args)