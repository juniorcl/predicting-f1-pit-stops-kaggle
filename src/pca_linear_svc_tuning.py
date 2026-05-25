#%%
import re
import sys
import optuna
import pickle
import logging

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder

from sklearn import set_config

from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler


#%%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pca_linear_svc.log'), 
        logging.StreamHandler(sys.stdout)
    ]
)

optuna_logger = optuna.logging.get_logger("optuna")
optuna_logger.handlers = logging.getLogger().handlers
optuna_logger.setLevel(logging.INFO)

set_config(transform_output="pandas")

def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

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
logging.info("----- Model Tuning -----")

def objective(trial, X, y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        X_train_fold = X.iloc[train_idx, :]
        X_valid_fold = X.iloc[valid_idx, :]

        y_train_fold = y.iloc[train_idx, 0]
        y_valid_fold = y.iloc[valid_idx, 0]

        model = make_pipeline(
            column_transformer,
            PCA(
                n_components=trial.suggest_float("n_components", 0.80, 0.99),
                svd_solver=trial.suggest_categorical("svd_solver", ["full"]),
                whiten=trial.suggest_categorical("whiten", [True, False]),
            ),
            StandardScaler(),
            LinearSVC(
                C=trial.suggest_float("C", 1e-5, 100, log=True),
                loss = trial.suggest_categorical("loss", ["squared_hinge"]),
                tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True),
                class_weight="balanced",
                dual=False,
                max_iter=10000,
                random_state=42
            )
        ).fit(X_train_fold, y_train_fold)

        score = model.decision_function(X_valid_fold)

        auc = roc_auc_score(y_valid_fold, score)
        aucs.append(auc)

        trial.report(np.mean(aucs), step=fold)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(aucs)

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100, n_jobs=-1, show_progress_bar=True)

logging.info(f"Best AUC: {study.best_value} | Best params: {study.best_params}")


#%%
logging.info("----- Saving Pipeline -----")

best_params = study.best_params

pca_keys = ["n_components", "svd_solver", "whiten"]

best_pca_params = {k: v for k, v in best_params.items() if k in pca_keys}
best_linear_svc_params = {k: v for k, v in best_params.items() if k not in pca_keys}

model_tuned = make_pipeline(
    column_transformer,
    PCA(**best_pca_params),
    StandardScaler(),
    LinearSVC(
        **best_linear_svc_params,
        class_weight="balanced",
        dual=False,
        max_iter=10000,
        random_state=42
    )
).fit(X_train, y_train.PitNextLap)


dump_pickle(model_tuned, '../models/model_pca_linear_svc.pkl')