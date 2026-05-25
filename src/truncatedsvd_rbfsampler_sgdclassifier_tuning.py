#%%
import sys
import optuna
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn import set_config
from category_encoders import CatBoostEncoder

from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.kernel_approximation import RBFSampler


#%%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/truncatedsvd_rbfsampler_sgdclassifier.log'), 
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
                n_components=trial.suggest_int("truncsvd_n_components", 2, 75),
                algorithm=trial.suggest_categorical("algorithm", ["randomized"]),
                n_iter=trial.suggest_int("n_iter", 3, 15),
                power_iteration_normalizer=trial.suggest_categorical("power_iteration_normalizer", ["auto", "OR", "LU"]),
                tol=trial.suggest_float("truncsvd_tol", 1e-6, 1e-2, log=True)
            ),
            StandardScaler(),
            RBFSampler(
                gamma=trial.suggest_float("gamma", 1e-6, 10.0, log=True),
                n_components=trial.suggest_int("n_components", 64, 75, log=True),
                random_state=42
            ),
            StandardScaler(),
            SGDClassifier(
                loss=trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
                penalty=trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
                alpha=trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
                learning_rate=trial.suggest_categorical("learning_rate", ["optimal", "adaptive"]),
                eta0=trial.suggest_float("eta0", 1e-4, 1e-1, log=True),
                l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                tol=trial.suggest_float("tol", 1e-5, 1e-2, log=True),
                average=trial.suggest_categorical("average", [True, False]),
                class_weight="balanced",
                max_iter=5000,
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
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50, n_jobs=-1, show_progress_bar=True)

logging.info(f"Best AUC: {study.best_value} | Best params: {study.best_params}")


#%%
logging.info("----- Saving Pipeline -----")

best_params = study.best_params

pipe_tuned = make_pipeline(
    column_transformer,
    TruncatedSVD(
        n_components=best_params["truncsvd_n_components"],
        algorithm=best_params["algorithm"],
        n_iter=best_params["n_iter"],
        power_iteration_normalizer=best_params["power_iteration_normalizer"],
        tol=best_params["truncsvd_tol"]
    ),
    StandardScaler(),
    RBFSampler(
        gamma=best_params["gamma"],
        n_components=best_params["n_components"],
        random_state=42
    ),
    StandardScaler(),
    SGDClassifier(
        loss=best_params["loss"],
        penalty=best_params["penalty"],
        alpha=best_params["alpha"],
        learning_rate=best_params["learning_rate"],
        eta0=best_params["eta0"],
        l1_ratio=best_params["l1_ratio"],
        tol=best_params["tol"],
        average=best_params["average"],
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
        n_jobs=1
    )
).fit(X_train, y_train.iloc[:, 0])


dump_pickle(pipe_tuned, '../models/model_truncatedsvd_rbfsampler_sgdclassifier.pkl')

# %%
