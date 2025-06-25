import pandas as pd
import os

# 读取数据
csv_path = 'report/trait_all_metrics.csv'
df = pd.read_csv(csv_path)

# 过滤掉CNN
df = df[df['Model'] != 'CNN']

# 计算每个性状、每个模型的平均PCC
pivot = df.groupby(['Trait', 'Model'])['PearsonR'].mean().reset_index()

# 对每个性状归一化PCC为权重
def normalize(group):
    total = group['PearsonR'].sum()
    group['Weight'] = group['PearsonR'] / total if total != 0 else 1.0 / len(group)
    return group

weights = pivot.groupby('Trait').apply(normalize)

# 只保留需要的列
weights = weights[['Trait', 'Model', 'Weight']]

# 输出目录
os.makedirs('report', exist_ok=True)
weights.to_csv('report/ensemble_weights_by_trait.csv', index=False)
print('权重表已保存到 report/ensemble_weights_by_trait.csv') 