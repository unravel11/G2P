# Wheat Genotype-Phenotype Machine Learning Pipeline

## 项目简介
本项目用于小麦基因型-表型预测，支持多性状、多模型（RandomForest、XGBoost、LightGBM、Lasso）、多SNP特征筛选、批量参数搜索与并行加速。适合大规模SNP数据的机器学习建模与特征重要性分析。

---

## 目录结构
```
├── data/                        # 数据目录
│   ├── wheat1k.pheno.txt       # 表型数据（样本×性状）
│   └── Wheat1k.recode.vcf      # 基因型VCF文件
├── src/                        # 源代码目录
│   ├── main.py                 # 主程序入口
│   ├── config.json            # 配置文件
│   ├── models/                # 模型实现
│   ├── utils/                 # 工具函数
│   │   ├── evaluation.py      # 模型评估
│   │   ├── training.py        # 训练相关
│   │   └── hyperparameter_tuning.py  # 参数调优
│   ├── data/                  # 数据处理
│   ├── visualization/         # 可视化
│   └── __init__.py
├── batch_experiment.py        # 批量实验脚本
├── requirements.txt           # 依赖包
├── output/                    # 结果输出目录
└── README.md
```

---

## 依赖环境
- Python 3.8+
- numpy, pandas, scikit-learn, matplotlib, seaborn
- xgboost, lightgbm
- tqdm, joblib, cyvcf2, scipy

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 数据格式说明
- **表型文件**：`data/wheat1k.pheno.txt`
  - 第一列为`sample`，后续为15个性状（列名为性状名）。
- **基因型文件**：`data/Wheat1k.recode.vcf`
  - VCF格式，样本与表型文件一致。

---

## 使用方法

### 1. 环境配置
首先安装所需依赖：
```bash
pip install -r requirements.txt
```

### 2. 数据准备
- 将表型数据文件 `wheat1k.pheno.txt` 放在 `data/` 目录下
- 将基因型数据文件 `Wheat1k.recode.vcf` 放在 `data/` 目录下

### 3. 配置文件说明
在 `src/config.json` 中可以配置：
- 数据文件路径
- 预处理参数（MAF阈值、缺失值阈值等）
- 模型参数（支持RandomForest、XGBoost、LightGBM、Lasso）
- 训练参数（测试集比例、交叉验证折数等）

### 4. 运行方式

#### 4.1 单性状多模型预测
```bash
# 方式1：使用run.py（推荐）
python run.py  # 使用默认配置

# 方式2：直接运行main.py
python src/main.py  # 使用默认配置

# 指定配置文件
python run.py --config path/to/config.json

# 指定特定模型和性状
python run.py --models RandomForest XGBoost --traits spikelength yield

# 启用参数搜索（使用配置文件中的参数网格）
python run.py --tune
```

> 注意：推荐使用 `run.py` 运行程序，它会自动处理Python路径问题，使用更方便。

### 5. 输出结果
- 模型评估结果保存在 `output/model_evaluation.txt`
- 特征重要性图保存在 `output/` 目录
- 预测效果图保存在 `output/` 目录

### 6. 结果解读
- `test_metrics`：测试集评估指标（R²、RMSE等）
- `cv_metrics`：交叉验证评估指标
- `top_features`：Top 10重要SNP位点
- 可视化结果包括：
  - 特征重要性条形图
  - 预测值vs实际值散点图

---

## 结果分析建议
- 关注`