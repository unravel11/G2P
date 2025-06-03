# Wheat Genotype-Phenotype Machine Learning Pipeline

## 项目简介
本项目用于小麦基因型-表型预测，支持多性状、多模型（RandomForest、XGBoost、LightGBM、Lasso）、多SNP特征筛选、批量参数搜索与并行加速。适合大规模SNP数据的机器学习建模与特征重要性分析。

---

## 目录结构
```
├── data/
│   ├── wheat1k.pheno.txt         # 表型数据（样本×性状）
│   └── Wheat1k.recode.vcf       # 基因型VCF文件
├── src/
│   ├── main.py                  # 单性状多模型主程序
│   ├── models/                  # 各类模型实现
│   ├── utils/                   # 评估、工具函数
│   └── ...
├── batch_experiment.py          # 批量实验与并行调参脚本
├── requirements.txt             # 依赖包
├── output/                      # 结果输出目录
│   └── batch_experiment_results.csv
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

## 单性状多模型建模
运行主程序：
```bash
python src/main.py
```
- 支持单性状多模型对比，自动输出评估结果、特征重要性、预测效果图。
- 结果保存在`output/model_evaluation.txt`。

---

## 批量实验与参数搜索（推荐）
批量遍历所有性状、所有模型、不同top_n_snps、所有参数组合，自动并行训练：
```bash
python batch_experiment.py
```
- 自动利用多核CPU并行加速（M1/M2/Intel/AMD均支持）。
- 结果保存在`output/batch_experiment_results.csv`，包含所有参数、评估指标、Top SNP。
- 可自定义参数网格，支持大规模调参。

---

## 结果分析建议
- 关注`test_r2`、`cv_r2`等指标，筛选最优模型与参数。
- 多模型Top SNP交集可用于生物学功能注释。
- 可用Excel/Pandas/可视化工具进一步分析`batch_experiment_results.csv`。

---

## 联系与贡献
如有问题或建议，欢迎提issue或PR。 