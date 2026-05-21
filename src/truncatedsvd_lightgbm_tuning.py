#%%
import sys
import optuna
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn import set_config
from lightgbm import LGBMClassifier
from category_encoders import CatBoostEncoder

from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

#%%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/truncatedsvd_lightgbm.log'), 
        logging.StreamHandler(sys.stdout)
    ]
)

optuna_logger = optuna.logging.get_logger("optuna")
optuna_logger.handlers = logging.getLogger().handlers
optuna_logger.setLevel(logging.INFO)


def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)


column_transformer = ColumnTransformer([
    (
        'target_encoder', 
        TargetEncoder(), 
        ['driver', 'compound', 'race']
    ),
    (
        'catboost_encoder', 
        CatBoostEncoder(), 
        ['driver', 'compound', 'race']
    ),
    (
        'standard_scaler', 
        StandardScaler(), 
        ['lapnumber', 'position', 'raceprogress', 'year', 'position_norm', 'race_progress_sin', 'position_vs_mean']
    ),
    (
        'robust_scaler', 
        RobustScaler(), 
        [
            'position_change', 'cumulative_degradation', 'laptime_delta', 'laptime_s', 'stint', 'driver_mean_lap', 'tyrelife', 'delta_x_tyre_life', 
            'compound_tyre_life', 'stint_progress', 'tyre_life_ratio', 'degradation_per_lap', 'position_change_cum', 'laps_since_pit', 'lap_time_inv',  
            'lap_time_vs_race_mean', 'lap_time_x_tyre', 'position_x_progress', 'degradation_x_progress', 'race_progress_squared', 'driver_avg_position' 
        ]
    ),
], remainder="passthrough")


#%%
X_train = pd.read_parquet('../data/processed/X_train.parquet')
y_train = pd.read_parquet('../data/processed/y_train.parquet')


#%%
logging.info("----- Fine Tuning -----")

def objective(trial, X, y):
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        
        X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx, 0], y.iloc[valid_idx, 0]

        model = make_pipeline(
            column_transformer,
            TruncatedSVD(
                n_components=trial.suggest_int("n_components", 2, 75),
                algorithm=trial.suggest_categorical("algorithm", ["randomized"]),
                n_iter=trial.suggest_int("n_iter", 3, 15),
                power_iteration_normalizer=trial.suggest_categorical("power_iteration_normalizer", ["auto", "OR", "LU"]),
                tol=trial.suggest_float("tol", 1e-6, 1e-2, log=True)
            ),
            LGBMClassifier(
                objective='binary',
                metric='auc',
                boosting_type='gbdt',
                num_leaves=trial.suggest_int('num_leaves', 16, 256),
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                lambda_l1=trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
                feature_fraction=trial.suggest_float('feature_fraction', 0.6, 1.0),
                bagging_fraction=trial.suggest_float('bagging_fraction', 0.6, 1.0),
                bagging_freq=trial.suggest_int('bagging_freq', 1, 7),
                min_child_samples=trial.suggest_int('min_child_samples', 10, 100),
                verbosity=-1,
                n_estimators=2000,
                random_state=42,
                n_jobs=1
            )
        ).fit(X_train, y_train)

        proba = model.predict_proba(X_valid)[:, 1]

        auc = roc_auc_score(y_valid, proba)
        aucs.append(auc)

        trial.report(np.mean(aucs), step=fold)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(aucs)

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30, n_jobs=-1, show_progress_bar=True)

logging.info(f"Best AUC: {study.best_value} | Best params: {study.best_params}")


#%%
logging.info("----- Saving Pipeline -----")

truncatedsvd_params = ["n_components", "algorithm", "n_iter", "power_iteration_normalizer", "tol"]

# best_params = {
#     'n_components': 59, 
#     'algorithm': 'randomized', 
#     'n_iter': 11, 
#     'power_iteration_normalizer': 'auto', 
#     'tol': 1.3022926982125576e-06, 
#     'num_leaves': 128, 
#     'max_depth': 10, 
#     'learning_rate': 0.03409600347639228, 
#     'lambda_l1': 1.0628563538033857, 
#     'lambda_l2': 7.868471599504208, 
#     'feature_fraction': 0.6626817767126574, 
#     'bagging_fraction': 0.8049311570827886, 
#     'bagging_freq': 2, 
#     'min_child_samples': 78
# }

best_truncatesvd_params = {k: v for k, v in best_params.items() if k in truncatedsvd_params}
best_lgbm_params = {k: v for k, v in best_params.items() if k not in truncatedsvd_params}

pipe_tuned = make_pipeline(
    column_transformer,
    TruncatedSVD(**best_truncatesvd_params),
    LGBMClassifier(
        **best_lgbm_params,
        verbosity=-1,
        n_estimators=2000,
        random_state=42,
        n_jobs=1
    )
).fit(X_train, y_train.iloc[:, 0])


dump_pickle(pipe_tuned, '../models/model_truncatedsvd_lightgbm.pkl')

# %%
