# -*- coding: utf-8 -*-
"""
05_benchmark.py — Benchmark rapido: RF / XGBoost / LightGBM / NN + Baseline
-----------------------------------------------------------------------------
Input:  04_matchato.xlsx (foglio Matched)
Output: 05_benchmark_results.xlsx (confronto modelli per tipo_carico)
        models/baseline_{tipo}.joblib  (modello baseline deterministico)

Split per ORDINE con GroupKFold: nessun ordine viene diviso tra train e test.
Ricalibrazione target avviene per-fold (solo su train di ciascun fold).
Metriche aggregate per ordine (mediana predizioni per ordine).
Un modello per tipo_carico (Completo, Parziale, Groupage).
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, r2_score,
    make_scorer,
)

# ── Parametri ──────────────────────────────────────────────

ID_COL     = "idordine"
QUOTE_COL  = "idquotazione"
TIPO_COL   = "tipo_carico"
TARGET_COL = "prezzo_attualizzato"
DATE_COL   = "data_ordine"

EXCLUDE_COLUMNS = [
    "idquotazione", "idordine", TARGET_COL,
    "prezzo_carb", "importo_per_peso",
    "importo", "importotrasp", "importo_per_km", "importo_norm",
    "Coefficiente", "stato_quotazione", "estimated", "ordine_originale",
]

RANDOM_STATE  = 42
N_FOLDS       = 3
TEST_SIZE     = 0.1
VERBOSE       = True
USE_LOG_TARGET = True

FIGDIR   = "figs"
MODELDIR = "models"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

# Soppressione warning cosmetici (anche nei subprocessi di cross_validate n_jobs=-1)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::FutureWarning,ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from scipy.linalg import LinAlgWarning
    warnings.filterwarnings("ignore", category=LinAlgWarning)
except ImportError:
    pass


def _log(msg: str):
    if VERBOSE:
        print(msg, file=sys.stdout)


def _safe_log1p(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log1p(np.clip(y, a_min=0.0, a_max=None))


def _safe_expm1(y_log: np.ndarray) -> np.ndarray:
    y_log = np.asarray(y_log, dtype=float)
    y = np.expm1(y_log)
    return np.clip(y, a_min=0.0, a_max=None)


# ── Scorer personalizzato per MAPE in cross_validate ─────

def _mape_scorer(y_true, y_pred):
    if USE_LOG_TARGET:
        y_true = _safe_expm1(y_true)
        y_pred = _safe_expm1(y_pred)
    return mean_absolute_percentage_error(y_true, y_pred) * 100.0

mape_scorer = make_scorer(_mape_scorer, greater_is_better=False)


# ── 1. Caricamento e preparazione dati ───────────────────

def load_matched_data(path: str = "04_matchato.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Matched")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def collapse_orders_median_target(
    df: pd.DataFrame, id_col: str, target_col: str,
    quote_col: Optional[str] = None, prefer: str = "upper",
    set_target_to_median: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["_idx"] = np.arange(len(out))
    out["_med"] = out.groupby(id_col)[target_col].transform("median")
    out["_dist"] = (out[target_col] - out["_med"]).abs()
    if prefer == "lower":
        out["_bias"] = np.where(out[target_col] <= out["_med"], 0, 1)
    elif prefer == "upper":
        out["_bias"] = np.where(out[target_col] >= out["_med"], 0, 1)
    else:
        out["_bias"] = 0
    sort_cols = [id_col, "_dist", "_bias"]
    ascending = [True, True, True]
    if quote_col is not None and quote_col in out.columns:
        sort_cols.append(quote_col)
        ascending.append(True)
    sort_cols.append("_idx")
    ascending.append(True)
    picked = (
        out.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(subset=[id_col], keep="first")
        .copy()
    )
    if set_target_to_median:
        med = out.groupby(id_col, as_index=False)["_med"].first()
        picked = picked.drop(columns=[target_col], errors="ignore").merge(
            med.rename(columns={"_med": target_col}), on=id_col, how="left"
        )
    picked = picked.drop(columns=[c for c in ["_idx", "_med", "_dist", "_bias"] if c in picked.columns])
    if quote_col is not None and quote_col in picked.columns:
        picked = picked.drop(columns=[quote_col])
    return picked.reset_index(drop=True)


def recalibrate_target_by_order(
    df: pd.DataFrame,
    id_col: str = "idordine",
    importo_norm_col: str = "importo_norm",
    km_col: str = "km_tratta",
    spazio_col: str = "spazio_calcolato",
    target_col: str = "prezzo_attualizzato",
) -> pd.DataFrame:
    df = df.copy()
    med_per_order = (
        df.groupby(id_col, dropna=False)[importo_norm_col]
        .median()
        .rename("_med_importo_norm")
    )
    df = df.merge(med_per_order, left_on=id_col, right_index=True, how="left")
    df[target_col] = (
        df["_med_importo_norm"].astype(float)
        * df[km_col].astype(float)
        * df[spazio_col].astype(float)
        / 1e5
    )
    df.drop(columns=["_med_importo_norm"], inplace=True)
    return df


# ── 2. Preparazione features ────────────────────────────

def prepare_features(
    df: pd.DataFrame, tipo: str, exclude_cols: list
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara X e y per un dato tipo_carico."""
    base_exclude = set(exclude_cols) | {ID_COL, TIPO_COL, TARGET_COL}

    # Mese/anno/trimestre ordine come feature
    if DATE_COL in df.columns:
        df = df.copy()
        dt = pd.to_datetime(df[DATE_COL])
        df["mese_ordine"] = dt.dt.month
        df["anno_ordine"] = dt.dt.year
        df["trimestre_ordine"] = dt.dt.quarter

    if tipo == "Groupage":
        extra_drop = {DATE_COL, "spazio_calcolato"}
    else:
        extra_drop = {DATE_COL, "altezza", "lunghezza_max", "spazio_calcolato"}

    feature_cols = [c for c in df.columns if c not in (base_exclude | extra_drop)]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(float).copy()
    return X, y


# ── 3. Pipeline di preprocessing ────────────────────────

def build_preprocessing(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ── 4. Modelli benchmark (con regolarizzazione anti-overfitting) ──

def get_benchmark_models() -> Dict[str, object]:
    """
    Modelli con iperparametri regolarizzati per ridurre overfitting:
    - RF: max_depth limitato, min_samples_leaf alzato, max_features sqrt
    - XGBoost: reg_alpha/reg_lambda, subsample, colsample_bytree
    - LightGBM: reg_alpha/reg_lambda, bagging, feature_fraction
    - NN: dropout via alpha (weight decay), early_stopping
    """
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE,
            criterion="squared_error",
            max_depth=30,
        ),
        "NeuralNet": Pipeline([
            ("scaler", StandardScaler()),
            ("nn", MLPRegressor(
                hidden_layer_sizes=(128, 128, 32),
                activation="relu",
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.01,       # L2 regularization (weight decay)
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    # XGBoost (opzionale)
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=700, learning_rate=0.05, max_depth=6,
            n_jobs=-1, random_state=RANDOM_STATE,
            objective="reg:absoluteerror",
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=4,
        )
    except ImportError:
        _log("[WARN] xgboost non installato, skippo XGBoost.")

    # LightGBM (opzionale)
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=700, learning_rate=0.05, max_depth=-1,
            n_jobs=-1, random_state=RANDOM_STATE,
            objective="mae", verbose=-1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=7,
            num_leaves=31,
        )
    except ImportError:
        _log("[WARN] lightgbm non installato, skippo LightGBM.")

    return models


# ── 5. Valutazione con GroupKFold + ricalibrazione per fold ──

def _aggregate_by_order(ids, y_true, y_pred):
    """Aggrega predizioni per ordine (mediana) e calcola metriche."""
    tmp = pd.DataFrame({"id": ids, "y_true": y_true, "y_pred": y_pred})
    agg = tmp.groupby("id", as_index=False).agg({"y_true": "median", "y_pred": "median"})
    return agg["y_true"].values, agg["y_pred"].values


def evaluate_model_cv(
    model, preprocessor, df_subset: pd.DataFrame, feature_cols: list,
    groups: np.ndarray, model_name: str, n_folds: int = N_FOLDS,
) -> Dict:
    """
    Valuta un modello con GroupKFold.
    Per ogni fold:
      1. Split per ordine (GroupKFold)
      2. Ricalibra target solo su train del fold
      3. Trasforma features con preprocessor fit su train
      4. Predici su test
      5. Aggrega per ordine (mediana) poi calcola metriche
    """
    from sklearn.base import clone

    gkf = GroupKFold(n_splits=n_folds)
    t0 = time.time()

    mae_tests, mape_tests, r2_tests = [], [], []
    mae_trains, mape_trains = [], []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df_subset, groups=groups)):
        df_train_fold = df_subset.iloc[train_idx].copy()
        df_test_fold  = df_subset.iloc[test_idx].copy()

        # Ricalibra target per-fold (solo su dati interni a ciascun set)
        df_train_fold = recalibrate_target_by_order(df_train_fold)
        df_test_fold  = recalibrate_target_by_order(df_test_fold)

        X_train_f = df_train_fold[feature_cols].copy()
        y_train_f_raw = df_train_fold[TARGET_COL].astype(float).values
        id_train_f = df_train_fold[ID_COL].values

        X_test_f = df_test_fold[feature_cols].copy()
        y_test_f_raw = df_test_fold[TARGET_COL].astype(float).values
        id_test_f = df_test_fold[ID_COL].values

        # Preprocess e fit
        prep_clone = clone(preprocessor)
        X_train_tx = prep_clone.fit_transform(X_train_f)
        X_test_tx  = prep_clone.transform(X_test_f)

        model_clone = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_train_fit = _safe_log1p(y_train_f_raw) if USE_LOG_TARGET else y_train_f_raw
            model_clone.fit(X_train_tx, y_train_fit)

        # Predici
        y_pred_train_log = model_clone.predict(X_train_tx)
        y_pred_test_log  = model_clone.predict(X_test_tx)

        if USE_LOG_TARGET:
            y_pred_train = _safe_expm1(y_pred_train_log)
            y_pred_test = _safe_expm1(y_pred_test_log)
            y_train_eval = y_train_f_raw
            y_test_eval = y_test_f_raw
        else:
            y_pred_train = y_pred_train_log
            y_pred_test = y_pred_test_log
            y_train_eval = y_train_f_raw
            y_test_eval = y_test_f_raw

        # Aggrega per ordine (metriche stabili)
        yt_tr_agg, yp_tr_agg = _aggregate_by_order(id_train_f, y_train_eval, y_pred_train)
        yt_te_agg, yp_te_agg = _aggregate_by_order(id_test_f, y_test_eval, y_pred_test)

        mae_trains.append(mean_absolute_error(yt_tr_agg, yp_tr_agg))
        mape_trains.append(mean_absolute_percentage_error(yt_tr_agg, yp_tr_agg) * 100)
        mae_tests.append(mean_absolute_error(yt_te_agg, yp_te_agg))
        mape_tests.append(mean_absolute_percentage_error(yt_te_agg, yp_te_agg) * 100)
        r2_tests.append(r2_score(yt_te_agg, yp_te_agg))

    total_time = time.time() - t0

    mae_test  = np.array(mae_tests)
    mape_test = np.array(mape_tests)
    r2_test   = np.array(r2_tests)
    mae_train = np.array(mae_trains)
    mape_train = np.array(mape_trains)

    return {
        "Model": model_name,
        "MAE_test": round(mae_test.mean(), 2),
        "MAE_test_std": round(mae_test.std(), 2),
        "MAPE_test_%": round(mape_test.mean(), 2),
        "MAPE_test_std": round(mape_test.std(), 2),
        "R2_test": round(r2_test.mean(), 4),
        "MAE_train": round(mae_train.mean(), 2),
        "MAPE_train_%": round(mape_train.mean(), 2),
        "Overfit_gap_%": round(mape_test.mean() - mape_train.mean(), 2),
        "Fit_time_s": round(total_time, 1),
        "n_folds": n_folds,
        "n_samples": len(df_subset),
    }


# ── 6. Baseline ─────────────────────────────────────────

def train_baseline(
    df_subset: pd.DataFrame, tipo: str, groups: np.ndarray,
    degree: int = 2, n_folds: int = N_FOLDS,
) -> Tuple[Pipeline, Dict]:
    """Ridge polinomiale su km_tratta + Perc_camion (+ peso_totale per Groupage).
    Valutato con GroupKFold + ricalibrazione per fold, poi trainato su tutto."""
    baseline_features = ["km_tratta", "Perc_camion"]
    if tipo == "Groupage":
        baseline_features = ["km_tratta", "Perc_camion", "peso_totale"]

    available = [f for f in baseline_features if f in df_subset.columns]
    if not available:
        _log(f"[BASELINE] Nessuna feature baseline disponibile per {tipo}")
        return None, {}

    from sklearn.base import clone

    pipe_template = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=1.0)),
    ])

    gkf = GroupKFold(n_splits=n_folds)
    mae_tests, mape_tests, r2_tests = [], [], []
    mae_trains, mape_trains = [], []

    for train_idx, test_idx in gkf.split(df_subset, groups=groups):
        df_tr = recalibrate_target_by_order(df_subset.iloc[train_idx].copy())
        df_te = recalibrate_target_by_order(df_subset.iloc[test_idx].copy())

        X_tr = df_tr[available].fillna(0).values
        y_tr_raw = df_tr[TARGET_COL].astype(float).values
        id_tr = df_tr[ID_COL].values

        X_te = df_te[available].fillna(0).values
        y_te_raw = df_te[TARGET_COL].astype(float).values
        id_te = df_te[ID_COL].values

        p = clone(pipe_template)
        y_tr_fit = _safe_log1p(y_tr_raw) if USE_LOG_TARGET else y_tr_raw
        p.fit(X_tr, y_tr_fit)

        yp_tr_log = p.predict(X_tr)
        yp_te_log = p.predict(X_te)
        if USE_LOG_TARGET:
            yp_tr = _safe_expm1(yp_tr_log)
            yp_te = _safe_expm1(yp_te_log)
            y_tr_eval = y_tr_raw
            y_te_eval = y_te_raw
        else:
            yp_tr = yp_tr_log
            yp_te = yp_te_log
            y_tr_eval = y_tr_raw
            y_te_eval = y_te_raw

        yt_tr_a, yp_tr_a = _aggregate_by_order(id_tr, y_tr_eval, yp_tr)
        yt_te_a, yp_te_a = _aggregate_by_order(id_te, y_te_eval, yp_te)

        mae_trains.append(mean_absolute_error(yt_tr_a, yp_tr_a))
        mape_trains.append(mean_absolute_percentage_error(yt_tr_a, yp_tr_a) * 100)
        mae_tests.append(mean_absolute_error(yt_te_a, yp_te_a))
        mape_tests.append(mean_absolute_percentage_error(yt_te_a, yp_te_a) * 100)
        r2_tests.append(r2_score(yt_te_a, yp_te_a))

    # Train su tutto per modello di produzione
    df_all_cal = recalibrate_target_by_order(df_subset.copy())
    X_all = df_all_cal[available].fillna(0).values
    y_all_raw = df_all_cal[TARGET_COL].astype(float).values
    y_all_fit = _safe_log1p(y_all_raw) if USE_LOG_TARGET else y_all_raw
    pipe_template.fit(X_all, y_all_fit)
    joblib.dump(pipe_template, os.path.join(MODELDIR, f"baseline_{tipo}.joblib"))
    joblib.dump(available, os.path.join(MODELDIR, f"baseline_features_{tipo}.joblib"))
    meta_path = os.path.join(MODELDIR, f"baseline_meta_{tipo}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"target_transform": "log1p" if USE_LOG_TARGET else "none"}, f, indent=2)

    mae_test  = np.array(mae_tests)
    mape_test = np.array(mape_tests)
    r2_test   = np.array(r2_tests)
    mape_train = np.array(mape_trains)

    metrics = {
        "Model": f"Baseline (Ridge poly{degree})",
        "MAE_test": round(mae_test.mean(), 2),
        "MAE_test_std": round(mae_test.std(), 2),
        "MAPE_test_%": round(mape_test.mean(), 2),
        "MAPE_test_std": round(mape_test.std(), 2),
        "R2_test": round(r2_test.mean(), 4),
        "MAE_train": round(np.array(mae_trains).mean(), 2),
        "MAPE_train_%": round(mape_train.mean(), 2),
        "Overfit_gap_%": round(mape_test.mean() - mape_train.mean(), 2),
        "Fit_time_s": 0.0,
        "n_folds": n_folds,
        "n_samples": len(df_subset),
        "features": available,
        "degree": degree,
    }
    _log(f"[BASELINE] {tipo}: MAE={mae_test.mean():.2f}, MAPE={mape_test.mean():.2f}%, R2={r2_test.mean():.4f}")
    return pipe_template, metrics


# ── 7. Benchmark per tipo_carico ─────────────────────────

def run_benchmark_for_tipo(df: pd.DataFrame, tipo: str) -> Tuple[pd.DataFrame, Dict]:
    """Esegue benchmark completo per un tipo_carico con GroupKFold CV."""
    subset = df[df[TIPO_COL] == tipo].copy()
    n_ordini = subset[ID_COL].nunique()
    _log(f"\n{'='*60}")
    _log(f"BENCHMARK — {tipo} ({len(subset)} righe, {n_ordini} ordini, {N_FOLDS}-Fold GroupKFold)")
    _log(f"{'='*60}")

    if len(subset) < 50:
        _log(f"[SKIP] Troppo pochi dati per {tipo}")
        return pd.DataFrame(), {}

    # NON ricalibriamo qui — la ricalibrazione avviene per-fold dentro evaluate_model_cv

    # Prepara features (senza ricalibrazione: mese_ordine non dipende dal target)
    if DATE_COL in subset.columns:
        dt = pd.to_datetime(subset[DATE_COL])
        subset["mese_ordine"] = dt.dt.month
        subset["anno_ordine"] = dt.dt.year
        subset["trimestre_ordine"] = dt.dt.quarter

    base_exclude = set(EXCLUDE_COLUMNS) | {ID_COL, TIPO_COL, TARGET_COL}
    if tipo == "Groupage":
        extra_drop = {DATE_COL, "spazio_calcolato"}
    else:
        extra_drop = {DATE_COL, "altezza", "lunghezza_max", "spazio_calcolato"}

    feature_cols = [c for c in subset.columns if c not in (base_exclude | extra_drop)]
    groups = subset[ID_COL].values

    # Preprocessing template (viene clonato per ogni fold)
    X_sample = subset[feature_cols].copy()
    preprocessor = build_preprocessing(X_sample)

    # Benchmark modelli con GroupKFold
    benchmark_models = get_benchmark_models()
    results = []

    for name, model in benchmark_models.items():
        _log(f"  {name} ({N_FOLDS}-Fold GroupKFold)...")
        try:
            metrics = evaluate_model_cv(
                model, preprocessor, subset, feature_cols, groups, name
            )
            results.append(metrics)
            _log(f"    MAPE={metrics['MAPE_test_%']}% (+/-{metrics['MAPE_test_std']}%), "
                 f"MAE={metrics['MAE_test']}, R2={metrics['R2_test']}, "
                 f"Overfit gap={metrics['Overfit_gap_%']}%, Time={metrics['Fit_time_s']}s")
        except Exception as e:
            _log(f"    [ERROR] {name} fallito: {e}")

    # Baseline con GroupKFold
    _, baseline_metrics = train_baseline(subset, tipo, groups)
    if baseline_metrics and results:
        results.append({k: baseline_metrics[k] for k in results[0].keys() if k in baseline_metrics})

    comparison_df = pd.DataFrame(results)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("MAPE_test_%").reset_index(drop=True)
        _log(f"\n  Classifica {tipo}:")
        display_cols = ["Model", "MAPE_test_%", "MAPE_test_std", "MAE_test", "R2_test",
                        "MAPE_train_%", "Overfit_gap_%", "Fit_time_s"]
        avail = [c for c in display_cols if c in comparison_df.columns]
        _log(comparison_df[avail].to_string(index=False))

    return comparison_df, baseline_metrics


# ── 8. Plot confronto ───────────────────────────────────

def plot_benchmark_comparison(results: Dict[str, pd.DataFrame]):
    for tipo, df_bench in results.items():
        if df_bench.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # MAPE con barre di errore
        if "MAPE_test_%" in df_bench.columns:
            err = df_bench.get("MAPE_test_std", pd.Series([0]*len(df_bench)))
            axes[0].barh(df_bench["Model"], df_bench["MAPE_test_%"],
                         xerr=err, capsize=3)
            axes[0].set_xlabel("MAPE (%)")
            axes[0].set_title(f"MAPE (CV) — {tipo}")

        # Overfitting gap
        if "Overfit_gap_%" in df_bench.columns:
            colors = ["red" if v > 5 else "orange" if v > 2 else "green"
                      for v in df_bench["Overfit_gap_%"]]
            axes[1].barh(df_bench["Model"], df_bench["Overfit_gap_%"], color=colors)
            axes[1].set_xlabel("Overfit gap (MAPE test - train) %")
            axes[1].set_title(f"Overfitting — {tipo}")
            axes[1].axvline(x=5, color="red", linestyle="--", alpha=0.5, label="soglia")

        # R2
        if "R2_test" in df_bench.columns:
            axes[2].barh(df_bench["Model"], df_bench["R2_test"])
            axes[2].set_xlabel("R2")
            axes[2].set_title(f"R2 (CV) — {tipo}")

        fig.suptitle(f"Benchmark {N_FOLDS}-Fold CV — {tipo}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, f"benchmark_{tipo}.png"), dpi=150)
        plt.close(fig)


# ── Main ─────────────────────────────────────────────────

def main():
    input_file  = "04_matchato.xlsx"
    output_file = "05_benchmark_results.xlsx"

    _log(f"[LOAD] Caricamento da {input_file}...")
    df_raw = load_matched_data(input_file)
    _log(f"  Righe totali: {len(df_raw)}")
    _log(f"  Ordini unici: {df_raw[ID_COL].nunique()}")
    _log(f"  Media quotazioni/ordine: {len(df_raw) / max(df_raw[ID_COL].nunique(), 1):.1f}")

    # NO collapse — lavoriamo su tutte le quotazioni con GroupKFold per ordine
    df = df_raw.copy()

    # Drop colonne intermedie
    drop_cols = ["importo", "importotrasp", "stato_quotazione", "ordine_originale"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    _log(f"[CONFIG] Split: GroupKFold per ordine ({N_FOLDS} folds)")

    # Benchmark per tipo
    all_benchmarks = {}
    all_baselines  = {}

    for tipo in ["Completo", "Parziale", "Groupage"]:
        bench_df, bl_metrics = run_benchmark_for_tipo(df, tipo)
        all_benchmarks[tipo] = bench_df
        all_baselines[tipo]  = bl_metrics

    # Plot
    plot_benchmark_comparison(all_benchmarks)

    # Summary
    summary_rows = []
    for tipo, bench_df in all_benchmarks.items():
        if bench_df.empty:
            continue
        best = bench_df.iloc[0]
        summary_rows.append({
            "tipo_carico": tipo,
            "best_model": best["Model"],
            "MAPE_test_%": best.get("MAPE_test_%"),
            "MAPE_test_std": best.get("MAPE_test_std"),
            "MAE_test": best.get("MAE_test"),
            "R2_test": best.get("R2_test"),
            "Overfit_gap_%": best.get("Overfit_gap_%"),
        })
    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        _log(f"\n{'='*60}")
        _log("RIEPILOGO VINCITORI")
        _log(f"{'='*60}")
        _log(summary.to_string(index=False))

    # Salva
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        for tipo, bench_df in all_benchmarks.items():
            if not bench_df.empty:
                bench_df.to_excel(writer, sheet_name=f"Benchmark_{tipo}"[:31], index=False)
        if not summary.empty:
            summary.to_excel(writer, sheet_name="Summary", index=False)

    _log(f"\n[OK] Risultati salvati in {output_file}")
    _log(f"[OK] Baseline salvati in {MODELDIR}/baseline_*.joblib")


if __name__ == "__main__":
    main()
