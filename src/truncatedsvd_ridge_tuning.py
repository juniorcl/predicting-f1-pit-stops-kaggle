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
from sklearn.inspection import permutation_importance
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

from feature_engine.selection import DropFeatures


#%%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/truncatedsvd_ridge.log'), 
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

    solver = trial.suggest_categorical("solver", ["auto", "cholesky", "lsqr", "sag"])
    params = {
        "alpha": trial.suggest_float("alpha", 1e-5, 100, log=True),
        "solver": solver,
        "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])
    }

    if solver in ["lsqr", "sag"]:
        params["max_iter"] = trial.suggest_int("max_iter", 500, 5000, step=500)

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
                tol=trial.suggest_float("truncsvd_tol", 1e-6, 1e-2, log=True)
            ),
            StandardScaler(),
            RidgeClassifier(**params, random_state=42)
        ).fit(X_train, y_train)

        decision = model.decision_function(X_valid)

        auc = roc_auc_score(y_valid, decision)
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

# best_params = study.best_params

best_params = {
    'solver': 'sag', 
    'alpha': 0.0012590930780247856, 
    'tol': 8.804610287179953e-05, 
    'fit_intercept': False, 
    'class_weight': 'balanced', 
    'max_iter': 1000, 
    'n_components': 75, 
    'algorithm': 'randomized', 
    'n_iter': 7, 
    'power_iteration_normalizer': 'LU', 
    'truncsvd_tol': 3.2634825986870737e-06
}

tuned_model = make_pipeline(
    column_transformer,
    TruncatedSVD(
        n_components=best_params["n_components"],
        algorithm=best_params["algorithm"],
        n_iter=best_params["n_iter"],
        power_iteration_normalizer=best_params["power_iteration_normalizer"],
        tol=best_params["truncsvd_tol"]
    ),
    StandardScaler(),
    RidgeClassifier(
        alpha=best_params["alpha"],
        solver=best_params["solver"],
        tol=best_params["tol"],
        fit_intercept=best_params["fit_intercept"],
        class_weight=best_params["class_weight"],
        max_iter=best_params["max_iter"],
        random_state=42
    )
).fit(X_train, y_train.iloc[:, 0])

dump_pickle(tuned_model, '../models/model_truncatedsvd_ridge.pkl')

# %%
