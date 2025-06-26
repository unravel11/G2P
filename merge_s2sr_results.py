import os
import re
import pandas as pd

# 1. 遍历所有性状目录
trait_dir = 'output/evaluate_20250626_140407'
report_file = 'report/trait_all_metrics.csv'
results = []

for trait in os.listdir(trait_dir):
    trait_path = os.path.join(trait_dir, trait)
    if not os.path.isdir(trait_path):
        continue
    result_file = os.path.join(trait_path, 'results.txt')
    if not os.path.exists(result_file):
        continue
    with open(result_file, encoding='utf-8') as f:
        txt = f.read()
        # 匹配所有S2SR模型结果
        for m in re.finditer(r'模型: S2SR_(\d+)\s+-+\s+测试集评估结果:\s+r2: ([\d\.-]+)\s+rmse: ([\d\.-]+)\s+mse: ([\d\.-]+)\s+pearson_r: ([\d\.-]+)', txt):
            snps, r2, rmse, mse, pearson = m.groups()
            results.append([
                trait, 'S2SR', int(snps), float(r2), float(rmse), float(pearson), result_file
            ])

# 2. 合并到原有csv
if results:
    df_new = pd.DataFrame(results, columns=['Trait','Model','SNPs','R2','RMSE','PearsonR','Source'])
    df_old = pd.read_csv(report_file)
    df_merged = pd.concat([df_old, df_new], ignore_index=True)
    df_merged.to_csv(report_file, index=False)
    print(f"合并完成，新增{len(df_new)}条S2SR结果。")
else:
    print("未发现S2SR结果。") 