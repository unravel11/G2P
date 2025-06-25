# G2P项目文件清单

## 项目结构概览

```
G2P/
├── src/                          # 源代码目录
│   ├── models/                   # 模型实现
│   │   ├── base.py              # 基础模型类
│   │   ├── lasso_model.py       # Lasso回归模型
│   │   ├── random_forest_model.py # 随机森林模型
│   │   ├── xgboost_model.py     # XGBoost模型
│   │   ├── lightgbm_model.py    # LightGBM模型
│   │   ├── ensemble_model.py    # 集成模型
│   │   └── factory.py           # 模型工厂
│   ├── data/                    # 数据处理
│   │   ├── preprocess_data.py   # 数据预处理
│   │   └── processed_data_loader.py # 数据加载器
│   ├── utils/                   # 工具函数
│   │   ├── training.py          # 训练工具
│   │   ├── evaluation.py        # 评估工具
│   │   └── ensemble_param_merger.py # 参数合并工具
│   ├── config.json              # 配置文件
│   └── main.py                  # 主程序入口
├── report/                      # 报告和结果
│   ├── trait_all_metrics.csv    # 所有实验结果
│   ├── G2P_Project_Report.md    # 完整项目报告
│   ├── G2P_Executive_Summary.md # 执行摘要
│   ├── Project_Files_Summary.md # 项目文件清单
│   └── figures/                 # 分析图表
│       ├── model_performance_analysis.png
│       └── ensemble_vs_others.png
├── output/                      # 输出结果
│   └── evaluate_20250625_112914/ # 最新评估结果
├── scripts/                     # 脚本文件
│   ├── update_metrics_table.py  # 更新结果表格
│   └── generate_report.py       # 生成报告
└── README.md                    # 项目说明
```

## 核心文件说明

### 1. 模型实现文件

#### `src/models/ensemble_model.py`
- **功能**：集成模型的核心实现
- **特色**：支持多种集成方法（平均、加权平均、投票、堆叠）
- **创新**：自动参数优化和嵌套参数处理

#### `src/models/factory.py`
- **功能**：模型工厂，统一创建和管理模型
- **特色**：支持配置文件驱动的模型创建
- **扩展性**：易于添加新模型

### 2. 工具函数

#### `src/utils/ensemble_param_merger.py`
- **功能**：自动合并集成参数和基础模型参数
- **创新**：嵌套参数空间处理
- **应用**：支持复杂的超参数优化

#### `src/utils/training.py`
- **功能**：模型训练和评估的统一接口
- **特色**：支持交叉验证和参数搜索
- **输出**：完整的评估结果和模型保存

### 3. 配置文件

#### `src/config.json`
- **功能**：项目配置中心
- **内容**：模型参数、数据路径、训练设置
- **特色**：支持嵌套参数定义

### 4. 数据文件

#### `report/trait_all_metrics.csv`
- **内容**：375个实验组合的完整结果
- **格式**：Trait, Model, SNPs, R2, RMSE, PearsonR, Source
- **价值**：为分析提供数据基础

### 5. 报告文件

#### `report/G2P_Project_Report.md`
- **内容**：完整的项目技术报告
- **结构**：背景、方法、实验、结果、结论
- **特色**：基于实际数据的深入分析

#### `report/G2P_Executive_Summary.md`
- **内容**：项目执行摘要
- **目标**：突出关键发现和贡献
- **受众**：项目管理和决策者

### 6. 分析图表

#### `report/figures/model_performance_analysis.png`
- **内容**：模型性能对比分析
- **包含**：性能排名、热图、SNP影响、分布图

#### `report/figures/ensemble_vs_others.png`
- **内容**：集成模型与其他模型对比
- **特色**：按性状和SNP数量的详细对比

## 实验数据统计

### 实验规模
- **性状数量**：15个
- **模型数量**：5个（Lasso, RandomForest, XGBoost, LightGBM, Ensemble）
- **SNP数量级别**：5个（100, 1000, 3000, 5000, 7000）
- **总实验组合**：375个

### 数据质量
- **数据完整性**：100%（无缺失值）
- **模型覆盖**：所有主流机器学习算法
- **评估指标**：R²、RMSE、Pearson相关系数

## 技术特色

### 1. 模块化设计
- **高内聚**：每个模块功能明确
- **低耦合**：模块间依赖最小化
- **易扩展**：新功能易于集成

### 2. 自动化程度高
- **配置驱动**：通过配置文件控制行为
- **批量处理**：支持多模型、多性状并行处理
- **结果管理**：自动保存和整理结果

### 3. 可重现性
- **随机种子固定**：确保结果可重现
- **参数记录**：完整记录实验参数
- **版本控制**：代码和配置版本化管理

## 使用指南

### 1. 环境配置
```bash
conda create -n g2p python=3.8
conda activate g2p
pip install -r requirements.txt
```

### 2. 运行实验
```bash
python src/main.py --config src/config.json --models Ensemble Lasso XGBoost --traits height yield --tune
```

### 3. 生成报告
```bash
python scripts/generate_report.py
```

### 4. 更新结果
```bash
python scripts/update_metrics_table.py
```

## 项目价值

### 1. 科学价值
- **方法学贡献**：创新的集成学习框架
- **基准数据集**：标准化的G2P评估基准
- **技术验证**：验证了多种机器学习方法在G2P中的应用

### 2. 应用价值
- **育种实践**：为分子标记辅助选择提供工具
- **研究平台**：为G2P研究提供标准化平台
- **教学资源**：机器学习在生物信息学中的应用案例

### 3. 技术价值
- **代码质量**：高质量的模块化代码
- **文档完善**：详细的技术文档和使用说明
- **可扩展性**：为后续研究提供坚实基础

## 维护和更新

### 1. 代码维护
- **定期更新**：保持依赖包的最新版本
- **bug修复**：及时修复发现的问题
- **性能优化**：持续改进算法性能

### 2. 功能扩展
- **新模型集成**：添加新的机器学习算法
- **数据源扩展**：支持更多作物和性状
- **功能增强**：添加可视化和分析功能

### 3. 文档更新
- **使用说明**：根据用户反馈更新文档
- **技术报告**：定期更新实验结果
- **最佳实践**：总结使用经验和建议

---

**项目维护者**：G2P开发团队  
**最后更新**：2025年1月  
**版本**：v1.0  
**许可证**：MIT License 