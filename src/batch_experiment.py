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
import json
import time
import numpy as np

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

def save_checkpoint(results, output_dir):
    """保存中间结果"""
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
    # 将结果转换为可序列化的格式
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        # 将numpy类型转换为Python原生类型
        if 'params' in serializable_result:
            serializable_result['params'] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                           for k, v in serializable_result['params'].items()}
        serializable_results.append(serializable_result)
    
    with open(checkpoint_file, 'w') as f:
        json.dump(serializable_results, f)
    logger.info(f"已保存检查点，当前完成 {len(results)} 个任务")

def load_checkpoint(output_dir):
    """加载已保存的结果"""
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            results = json.load(f)
        logger.info(f"已加载检查点，包含 {len(results)} 个已完成任务")
        return results
    return []

def get_completed_tasks(results):
    """获取已完成任务的标识"""
    completed = set()
    for result in results:
        if 'error' not in result:  # 只考虑成功完成的任务
            key = (result['trait'], result['model'], result['top_n_snps'], 
                  json.dumps(result['params'], sort_keys=True))
            completed.add(key)
    return completed

if __name__ == '__main__':
    # 数据加载
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pheno_file = os.path.join(project_root, 'data', 'wheat1k.pheno.txt')
    geno_file = os.path.join(project_root, 'data', 'Wheat1k.recode.vcf')
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 加载检查点
    results = load_checkpoint(output_dir)
    completed_tasks = get_completed_tasks(results)

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
                    # 检查任务是否已完成
                    task_key = (trait, model_name, top_n, json.dumps(param_set, sort_keys=True))
                    if task_key not in completed_tasks:
                        tasks.append((trait, top_n, model_name, ModelClass, param_set, extra_kwargs,
                                    genotype_matrix, snp_ids, sample_ids, pheno_data))

    logger.info(f"总任务数: {len(tasks) + len(completed_tasks)}, 已完成: {len(completed_tasks)}, 待完成: {len(tasks)}")

    if not tasks:
        logger.info("所有任务已完成！")
    else:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(run_single_experiment, *task) for task in tasks]
            last_save_time = time.time()
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 每5分钟保存一次检查点
                    current_time = time.time()
                    if current_time - last_save_time > 300:  # 300秒 = 5分钟
                        save_checkpoint(results, output_dir)
                        last_save_time = current_time
                        
                except Exception as e:
                    logger.error(f"任务执行出错: {e}")

        # 保存最终结果
        save_checkpoint(results, output_dir)
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'batch_experiment_results.csv'), index=False)
        logger.info('所有实验已完成，结果已保存到 output/batch_experiment_results.csv') 