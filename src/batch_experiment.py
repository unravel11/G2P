import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import RandomForestModel, XGBoostModel, LightGBMModel, LassoModel
from data.loader import DataLoader
from data.preprocessor import GenotypePreprocessor
from utils.evaluation import evaluate_model, cross_validate
import logging
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 参数网格（扩展搜索空间）
top_snp_list = [1000, 2000, 5000]
rf_param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'random_state': [42]
}
xgb_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 1, 10],
    'random_state': [42]
}
lgb_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 1, 10],
    'random_state': [42],
    'verbose': [-1]
}
lasso_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'max_iter': [10000],
    'random_state': [42]
}

def param_product(param_grid):
    keys = list(param_grid.keys())
    for values in product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, values))

model_grid = {
    'RandomForest': (RandomForestModel, param_product(rf_param_grid), {'task_type': 'regression'}),
    'XGBoost': (XGBoostModel, param_product(xgb_param_grid), {'task_type': 'regression'}),
    'LightGBM': (LightGBMModel, param_product(lgb_param_grid), {'task_type': 'regression'}),
    'Lasso': (LassoModel, param_product(lasso_param_grid), {}),
}

def run_single_experiment(trait, top_n, model_name, ModelClass, param_set, extra_kwargs,
                          genotype_matrix, snp_ids, sample_ids, pheno_data):
    try:
        y_all = pheno_data[trait].values
        preprocessor = GenotypePreprocessor(
            maf_threshold=0.05,
            missing_threshold=0.1,
            gwas_p_threshold=1e-5,
            top_n_snps=top_n
        )
        filtered_matrix, filtered_snp_ids, filtered_sample_ids = preprocessor.preprocess(
            genotype_matrix, snp_ids, sample_ids, y_all
        )
        pheno_trait = pheno_data.loc[filtered_sample_ids]
        y = pheno_trait[trait].values
        scaler = StandardScaler()
        X = scaler.fit_transform(filtered_matrix.T)
        X_df = pd.DataFrame(X, columns=filtered_snp_ids)
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )
        kwargs = dict(model_params=param_set, **extra_kwargs)
        model = ModelClass(**kwargs)
        model.train(X_train, y_train, feature_names=filtered_snp_ids)
        y_pred = model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred)
        mean_metrics, fold_metrics = cross_validate(model, X_df, y, n_splits=5)
        feature_importance = model.get_feature_importance()
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            'trait': trait,
            'model': model_name,
            'top_n_snps': top_n,
            'params': param_set,
            'test_r2': test_metrics.get('r2', None),
            'test_rmse': test_metrics.get('rmse', None),
            'cv_r2': mean_metrics.get('r2', None),
            'cv_rmse': mean_metrics.get('rmse', None),
            'top_features': top_features
        }
    except Exception as e:
        return {'trait': trait, 'model': model_name, 'top_n_snps': top_n, 'params': param_set, 'error': str(e)}

if __name__ == '__main__':
    # 数据加载
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pheno_file = os.path.join(project_root, 'data', 'wheat1k.pheno.txt')
    geno_file = os.path.join(project_root, 'data', 'Wheat1k.recode.vcf')
    loader = DataLoader()
    pheno_data = loader.load_phenotype(pheno_file).set_index('sample')
    genotype_matrix, snp_ids, sample_ids, chroms = loader.load_genotype(geno_file)

    # 样本匹配
    common_samples = list(set(sample_ids) & set(pheno_data.index))
    pheno_data = pheno_data.loc[common_samples]
    sample_indices = [sample_ids.index(s) for s in common_samples]
    genotype_matrix = genotype_matrix[:, sample_indices]
    sample_ids = [sample_ids[i] for i in sample_indices]

    trait_list = pheno_data.columns.tolist()
    tasks = []
    for trait in trait_list:
        for top_n in top_snp_list:
            for model_name, (ModelClass, param_iter, extra_kwargs) in model_grid.items():
                for param_set in param_iter:
                    tasks.append((trait, top_n, model_name, ModelClass, param_set, extra_kwargs,
                                  genotype_matrix, snp_ids, sample_ids, pheno_data))

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_single_experiment, *task) for task in tasks]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in task: {e}")

    # 保存结果
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'batch_experiment_results.csv'), index=False)
    print('所有实验已完成，结果已保存到 output/batch_experiment_results.csv') 