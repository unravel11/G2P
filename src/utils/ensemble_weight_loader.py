import pandas as pd

def load_trait_weights(weight_csv_path):
    """
    读取每个性状的Ensemble权重表，返回字典 {trait: [lasso_weight, lightgbm_weight, xgboost_weight]}
    """
    df = pd.read_csv(weight_csv_path)
    model_order = ['lasso', 'lightgbm', 'xgboost']
    trait_weights = {}
    for trait, group in df.groupby('Trait'):
        weights = []
        for m in model_order:
            row = group[group['Model'].str.lower() == m]
            if not row.empty:
                weights.append(row['Weight'].values[0])
            else:
                weights.append(1.0 / len(model_order))
        trait_weights[trait] = weights
    return trait_weights 