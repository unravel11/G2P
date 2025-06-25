import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取数据
csv_path = 'report/trait_all_metrics.csv'
df = pd.read_csv(csv_path)

# 过滤掉CNN
df = df[df['Model'] != 'CNN']

# 计算每个性状、每个模型的平均R²
pivot = df.groupby(['Trait', 'Model'])['R2'].mean().reset_index()

# 找到每个性状下R²最高的模型
best = pivot.loc[pivot.groupby('Trait')['R2'].idxmax()]

# 画图
plt.figure(figsize=(18, 8))
sns.set(style="whitegrid", font_scale=1.2)

# 分组柱状图
ax = sns.barplot(
    data=pivot,
    x='Trait', y='R2', hue='Model',
    palette='Set2',
    edgecolor='black'
)

# 标注最优模型
for i, trait in enumerate(pivot['Trait'].unique()):
    best_row = best[best['Trait'] == trait]
    if not best_row.empty:
        best_model = best_row['Model'].values[0]
        best_r2 = best_row['R2'].values[0]
        # 找到该模型在hue中的位置
        model_list = pivot['Model'].unique().tolist()
        model_idx = model_list.index(best_model)
        # 计算柱子的x坐标
        bar_pos = i + model_idx / len(model_list) - 0.2
        ax.text(i, best_r2 + 0.01, f"★{best_model}", color='red', ha='center', fontsize=10, fontweight='bold')

plt.title('每个性状 × 每个模型的R²性能（不含CNN）', fontsize=16)
plt.ylabel('平均R²')
plt.xlabel('性状')
plt.ylim(0, 1)
plt.legend(title='模型', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()

# 输出目录
os.makedirs('report/figures', exist_ok=True)
plt.savefig('report/figures/trait_model_performance.png', dpi=300)
plt.close()
print('图已保存到 report/figures/trait_model_performance.png') 