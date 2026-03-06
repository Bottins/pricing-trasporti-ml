# -*- coding: utf-8 -*-
"""
06_training.py — Addestramento serio del modello migliore per tipo_carico
--------------------------------------------------------------------------
Input:  04_matchato.xlsx (foglio Matched)
Output: 06_training_results.xlsx
        models/model_{tipo}.joblib
        models/features_{tipo}.txt
        models/metadata_{tipo}.json
        models/gmm_{tipo}.joblib

Modello per tipo_carico scelto dal benchmark:
  - Completo:  LightGBM
  - Parziale:  LightGBM
  - Groupage:  LightGBM

Split per ORDINE (GroupShuffleSplit): tutte le quotazioni di uno stesso ordine
finiscono interamente in train OPPURE in test — nessun data leakage.
La ricalibrazione target avviene solo sul train set.
Le metriche su test sono aggregate per ordine (mediana predizioni per ordine).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from typing import Dict, List, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.mixture import GaussianMixture

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

# ── Scelta modello per tipo_carico (da benchmark) ─────────
#    Cambia qui se un nuovo benchmark indica un vincitore diverso.

MODEL_PER_TIPO = {
    "Completo": "LightGBM",
    "Parziale": "LightGBM",
    "Groupage": "LightGBM",
}

# ── Hyperparameters generali ──────────────────────────────

RANDOM_STATE       = 1
TEST_SIZE          = 0.02
INTERVAL_PERCENT   = 50.0
SCALE_METHOD       = "gmm_global"
MAX_GMM_COMPONENTS = 4

# ── Hyperparameters LightGBM ─────────────────────────────

LGBM_PARAMS = {
    "n_estimators":      750,
    "learning_rate":     0.05,
    "max_depth":         -1,
    "num_leaves":        31,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "min_child_samples": 5,
    "objective":         "mape",
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      RANDOM_STATE,
}

# Target transform / tuning / feature selection
USE_LOG_TARGET = True
USE_OPTUNA = False
OPTUNA_TRIALS = 20
FEATURE_SELECTION = True
FEATURE_MIN_IMPORTANCE_RATIO = 0.001  # 0.1% of total importance

# Feature protette dalla feature selection (categoriche one-hot hanno importanza frammentata)
FEATURE_WHITELIST = {
    "macro_carico", "macro_scarico",
    "is_sardegna", "is_sicilia", "is_fuori_misura",
    "verso_est", "verso_nord",
    "anno_ordine", "trimestre_ordine",
    "km_x_perc", "km_x_verso_nord",
}

# ── Hyperparameters XGBoost ──────────────────────────────

XGB_PARAMS = {
    "n_estimators":      750,
    "learning_rate":     0.05,
    "max_depth":         6,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "objective":         "reg:absoluteerror",
    "n_jobs":            -1,
    "random_state":      RANDOM_STATE,
}

VERBOSE    = True
FIGDIR     = "figs"
MODELDIR   = "models"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

# Soppressione warning cosmetici
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _log(msg: str):
    if VERBOSE:
        print(msg, file=sys.stdout)


def _savefig(filename: str):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), dpi=200)
    plt.close()


def _safe_log1p(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log1p(np.clip(y, a_min=0.0, a_max=None))


def _safe_expm1(y_log: np.ndarray) -> np.ndarray:
    y_log = np.asarray(y_log, dtype=float)
    y = np.expm1(y_log)
    return np.clip(y, a_min=0.0, a_max=None)


# ── 1. Caricamento e preparazione ────────────────────────

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
    if quote_col and quote_col in out.columns:
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
    if quote_col and quote_col in picked.columns:
        picked = picked.drop(columns=[quote_col])
    return picked.reset_index(drop=True)


def recalibrate_target_by_order(
    df: pd.DataFrame, id_col: str = "idordine",
    importo_norm_col: str = "importo_norm", km_col: str = "km_tratta",
    spazio_col: str = "spazio_calcolato", target_col: str = "prezzo_attualizzato",
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


# ── 2. Creazione estimatore per tipo ─────────────────────

def _build_estimator(tipo: str):
    """Restituisce l'estimatore corretto per il tipo_carico."""
    model_name = MODEL_PER_TIPO.get(tipo, "XGBoost")

    if model_name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**LGBM_PARAMS), model_name

    elif model_name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBRegressor(**XGB_PARAMS), model_name

    else:
        raise ValueError(f"Modello '{model_name}' non supportato. Usa LightGBM o XGBoost.")


# ── 3. Pipeline ─────────────────────────────────────────

def build_pipeline(X: pd.DataFrame, estimator=None) -> Pipeline:
    if estimator is None:
        raise ValueError("Devi passare un estimator esplicito.")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([("prep", preprocessor), ("model", estimator)])


# ── 4. Feature importance e incertezza ───────────────────

def get_feature_importances(pipe: Pipeline) -> pd.DataFrame:
    preproc = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    try:
        names = preproc.get_feature_names_out()
    except Exception:
        names = np.array([f"f{i}" for i in range(len(model.feature_importances_))], dtype=object)
    imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def show_feature_importances(pipe: Pipeline, tipo: str, top_n: int = 10):
    df_imp = get_feature_importances(pipe)
    _log(f"\n  Feature importances — {tipo} (top {top_n}):")
    _log(df_imp.head(top_n).to_string(index=False))
    top = df_imp.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, max(4, 0.35 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Feature Importances — {tipo} (top {top_n})")
    _savefig(f"feature_importances_{tipo}.png")


def per_prediction_uncertainty(model, X_tx, interval_percent, model_name: str, pred_transform=None):
    """
    Calcola incertezza per-predizione.
    Per XGBoost/LightGBM: usa predizioni per iterazione (ntree_limit crescenti).
    Per RF: usa i singoli alberi.
    """
    lower_q = (100 - interval_percent) / 2
    upper_q = 100 - lower_q

    use_last = model_name in ["LightGBM", "XGBoost"]
    if model_name == "LightGBM":
        # LightGBM: predici con numero crescente di iterazioni
        n_iter = model.n_estimators_  # iterazioni effettive dopo il fit
        # Campiona ~50 checkpoint equidistanti per efficienza
        checkpoints = np.unique(np.linspace(1, n_iter, min(50, n_iter), dtype=int))
        preds_at_checkpoints = np.stack([
            model.predict(X_tx, num_iteration=k)
            for k in checkpoints
        ], axis=1)
        preds = preds_at_checkpoints

    elif model_name == "XGBoost":
        # XGBoost: predici con numero crescente di iterazioni
        n_iter = model.get_booster().num_boosted_rounds()
        checkpoints = np.unique(np.linspace(1, n_iter, min(50, n_iter), dtype=int))
        preds_at_checkpoints = np.stack([
            model.predict(X_tx, iteration_range=(0, k))
            for k in checkpoints
        ], axis=1)
        preds = preds_at_checkpoints

    else:
        # RF: usa i singoli alberi
        preds = np.stack([est.predict(X_tx) for est in model.estimators_], axis=1)

    if pred_transform is not None:
        preds = pred_transform(preds)

    mean_pred = preds[:, -1] if use_last else np.mean(preds, axis=1)
    std_pred = np.std(preds, axis=1, ddof=1)
    lower = np.percentile(preds, lower_q, axis=1)
    upper = np.percentile(preds, upper_q, axis=1)

    return mean_pred, std_pred, np.stack([lower, upper], axis=1)


# ── 5. GMM per incertezza ───────────────────────────────

def gmm_fit_on_target(y, max_components=MAX_GMM_COMPONENTS, random_state=0):
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    models, bics = [], []
    for k in range(1, max_components + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gm.fit(y)
        models.append(gm)
        bics.append(gm.bic(y))
    return models[int(np.argmin(bics))]


def gmm_mixture_std(gm):
    w = gm.weights_
    mu = gm.means_.flatten()
    sigma2 = np.array([np.squeeze(gm.covariances_[i]) for i in range(gm.n_components)], dtype=float)
    mu_mix = np.sum(w * mu)
    var_mix = np.sum(w * (sigma2 + (mu - mu_mix) ** 2))
    return float(np.sqrt(max(var_mix, 1e-12)))


def confidence_from_scale(std_pred, scale):
    eps = 1e-9
    denom = np.asarray(std_pred, dtype=float) + np.asarray(scale, dtype=float) + eps
    conf = 1.0 - (std_pred / denom)
    return np.clip(conf, 0.0, 1.0)


# ── 6. Salvataggio modello ──────────────────────────────

def save_model_bundle(tipo, pipe, feature_cols, meta, gmm=None):
    model_path = os.path.join(MODELDIR, f"model_{tipo}.joblib")
    joblib.dump(pipe, model_path)

    features_path = os.path.join(MODELDIR, f"features_{tipo}.txt")
    with open(features_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    meta_path = os.path.join(MODELDIR, f"metadata_{tipo}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if gmm is not None:
        gmm_path = os.path.join(MODELDIR, f"gmm_{tipo}.joblib")
        joblib.dump(gmm, gmm_path)

    _log(f"  [MODEL] Salvati: {model_path}, {features_path}, {meta_path}")


def _aggregate_base_importance(imp_df: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
    base_importance = {f: 0.0 for f in base_features}
    for feat, imp in zip(imp_df["feature"].values, imp_df["importance"].values):
        matched = False
        for base in base_features:
            if feat == base or feat.startswith(f"{base}_"):
                base_importance[base] += float(imp)
                matched = True
                break
        if not matched and feat in base_importance:
            base_importance[feat] += float(imp)
    out = pd.DataFrame({"feature": list(base_importance.keys()), "importance": list(base_importance.values())})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def _select_features_by_importance(X_train: pd.DataFrame, y_train, min_ratio: float) -> List[str]:
    try:
        import lightgbm as lgb
    except ImportError:
        _log("  [WARN] lightgbm non installato: salto feature selection.")
        return X_train.columns.tolist()

    quick_params = dict(LGBM_PARAMS)
    quick_params["n_estimators"] = min(300, LGBM_PARAMS.get("n_estimators", 750))
    estimator = lgb.LGBMRegressor(**quick_params)
    pipe = build_pipeline(X_train, estimator=estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)

    imp_df = get_feature_importances(pipe)
    base_imp = _aggregate_base_importance(imp_df, X_train.columns.tolist())
    total = base_imp["importance"].sum()
    if total <= 0:
        return X_train.columns.tolist()

    keep = base_imp[base_imp["importance"] >= (min_ratio * total)]["feature"].tolist()

    # Forza whitelist: categoriche one-hot hanno importanza frammentata per design
    rescued = []
    for wl in FEATURE_WHITELIST:
        if wl in X_train.columns and wl not in keep:
            keep.append(wl)
            rescued.append(wl)

    dropped = [c for c in X_train.columns if c not in keep]
    if rescued:
        _log(f"  [FS] Whitelist: salvate {len(rescued)} feature: {rescued}")
    if dropped:
        _log(f"  [FS] Droppate {len(dropped)} feature (importanza < {min_ratio*100:.2f}%): {dropped}")
    return keep


def _optuna_tune(
    X_train: pd.DataFrame,
    y_train,
    groups: np.ndarray,
    n_trials: int = OPTUNA_TRIALS,
) -> Dict:
    try:
        import optuna
        import lightgbm as lgb
    except ImportError:
        _log("  [WARN] optuna/lightgbm non installati: salto tuning.")
        return {}

    from sklearn.model_selection import GroupKFold

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "num_leaves": trial.suggest_int("num_leaves", 10, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "objective": "mae",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }

        gkf = GroupKFold(n_splits=3)
        mape_scores = []
        for tr_idx, va_idx in gkf.split(X_train, groups=groups):
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_tr = y_train[tr_idx]
            y_va = y_train[va_idx]

            estimator = lgb.LGBMRegressor(**params)
            pipe = build_pipeline(X_tr, estimator=estimator)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_tr, y_tr)

            y_pred_log = pipe.predict(X_va)
            if USE_LOG_TARGET:
                y_pred = _safe_expm1(y_pred_log)
                y_true = _safe_expm1(y_va)
            else:
                y_pred = y_pred_log
                y_true = y_va

            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            mape_scores.append(mape)

        return float(np.mean(mape_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    _log(f"  [OPTUNA] Best MAPE={study.best_value:.3f}%")
    return study.best_params


# ── 7. Training per tipo ────────────────────────────────

def recalibrate_target_train_only(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    id_col: str = "idordine", importo_norm_col: str = "importo_norm",
    km_col: str = "km_tratta", spazio_col: str = "spazio_calcolato",
    target_col: str = "prezzo_attualizzato",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ricalibra il target usando la mediana di importo_norm per ordine.
    La mediana è calcolata SOLO sul train set per evitare data leakage.
    Per il test set: la mediana viene calcolata internamente al test (ordini
    completamente separati) — nessun leakage perché gli ordini non si sovrappongono.
    """
    df_train = recalibrate_target_by_order(
        df_train, id_col, importo_norm_col, km_col, spazio_col, target_col
    )
    df_test = recalibrate_target_by_order(
        df_test, id_col, importo_norm_col, km_col, spazio_col, target_col
    )
    return df_train, df_test


def aggregate_predictions_by_order(
    id_values: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggrega predizioni per ordine (mediana) per metriche stabili.
    Ritorna (id_ordine_unici, y_true_agg, y_pred_agg).
    """
    tmp = pd.DataFrame({"id": id_values, "y_true": y_true, "y_pred": y_pred})
    agg = tmp.groupby("id", as_index=False).agg({"y_true": "median", "y_pred": "median"})
    return agg["id"].values, agg["y_true"].values, agg["y_pred"].values


def fit_and_evaluate_by_tipo(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    results_summary = {}
    predictions_per_tipo = {}
    base_exclude = set(EXCLUDE_COLUMNS) | {ID_COL, TIPO_COL, TARGET_COL}

    for tipo in ["Completo", "Parziale", "Groupage"]:
        subset = df[df[TIPO_COL] == tipo].copy()

        estimator, model_name = _build_estimator(tipo)
        model_params_used = None

        n_ordini = subset[ID_COL].nunique()
        _log(f"\n{'='*60}")
        _log(f"TRAINING — {tipo} ({len(subset)} righe, {n_ordini} ordini) — {model_name}")
        _log(f"{'='*60}")

        if len(subset) < 50:
            _log(f"  [SKIP] Troppo pochi dati per {tipo}")
            continue

        # Feature (prima della ricalibrazione: mese_ordine non dipende dal target)
        if DATE_COL in subset.columns:
            dt = pd.to_datetime(subset[DATE_COL])
            subset["mese_ordine"] = dt.dt.month
            subset["anno_ordine"] = dt.dt.year
            subset["trimestre_ordine"] = dt.dt.quarter

        if all(c in subset.columns for c in ["km_tratta", "Perc_camion"]):
            subset["km_x_perc"] = subset["km_tratta"].astype(float) * subset["Perc_camion"].astype(float)
        if all(c in subset.columns for c in ["km_tratta", "verso_nord"]):
            subset["km_x_verso_nord"] = subset["km_tratta"].astype(float) * subset["verso_nord"].abs().astype(float)

        if tipo == "Groupage":
            extra_drop = {DATE_COL, "spazio_calcolato"}
        else:
            extra_drop = {DATE_COL, "altezza", "lunghezza_max", "spazio_calcolato"}

        feature_cols = [c for c in subset.columns if c not in (base_exclude | extra_drop)]

        # ── Split per ORDINE (GroupShuffleSplit) ──────────────
        groups = subset[ID_COL].values
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(subset, groups=groups))

        df_train_raw = subset.iloc[train_idx].copy()
        df_test_raw  = subset.iloc[test_idx].copy()

        # Verifica: nessun ordine condiviso tra train e test
        ordini_train = set(df_train_raw[ID_COL].unique())
        ordini_test  = set(df_test_raw[ID_COL].unique())
        overlap = ordini_train & ordini_test
        assert len(overlap) == 0, f"LEAKAGE! {len(overlap)} ordini in entrambi train/test"
        _log(f"  Split per ordine: {len(ordini_train)} ordini train, {len(ordini_test)} ordini test")
        _log(f"  Righe: train={len(df_train_raw)}, test={len(df_test_raw)}")

        # ── Ricalibra target SOLO con dati interni a ciascun set ──
        df_train_cal, df_test_cal = recalibrate_target_train_only(
            df_train_raw, df_test_raw
        )

        X_train = df_train_cal[feature_cols].copy()
        y_train_raw = df_train_cal[TARGET_COL].astype(float).copy().values
        id_train = df_train_cal[ID_COL].values

        X_test = df_test_cal[feature_cols].copy()
        y_test_raw = df_test_cal[TARGET_COL].astype(float).copy().values
        id_test = df_test_cal[ID_COL].values

        if USE_LOG_TARGET:
            y_train = _safe_log1p(y_train_raw)
            y_test = _safe_log1p(y_test_raw)
        else:
            y_train = y_train_raw
            y_test = y_test_raw

        if FEATURE_SELECTION:
            kept = _select_features_by_importance(X_train, y_train, FEATURE_MIN_IMPORTANCE_RATIO)
            X_train = X_train[kept].copy()
            X_test = X_test[kept].copy()
            feature_cols = kept

        if USE_OPTUNA and model_name == "LightGBM":
            best_params = _optuna_tune(X_train, y_train, groups=df_train_cal[ID_COL].values)
            if best_params:
                import lightgbm as lgb
                tuned_params = dict(LGBM_PARAMS)
                tuned_params.update(best_params)
                estimator = lgb.LGBMRegressor(**tuned_params)
                model_params_used = {k: v for k, v in tuned_params.items() if k != "verbose"}
        if model_name == "LightGBM" and model_params_used is None:
            model_params_used = {k: v for k, v in LGBM_PARAMS.items() if k != "verbose"}

        # Build & train
        pipe = build_pipeline(X_train, estimator=estimator)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)

        show_feature_importances(pipe, tipo)

        # ── Metriche TRAIN (aggregate per ordine) ─────────────
        y_pred_train_log = pipe.predict(X_train)
        if USE_LOG_TARGET:
            y_pred_train = _safe_expm1(y_pred_train_log)
            y_train_eval = y_train_raw
        else:
            y_pred_train = y_pred_train_log
            y_train_eval = y_train
        _, y_tr_agg, yp_tr_agg = aggregate_predictions_by_order(id_train, y_train_eval, y_pred_train)
        mae_train  = mean_absolute_error(y_tr_agg, yp_tr_agg)
        mape_train = mean_absolute_percentage_error(y_tr_agg, yp_tr_agg) * 100
        _log(f"  [TRAIN] MAE={mae_train:.2f}, MAPE={mape_train:.2f}% (aggregato per ordine)")

        # ── Metriche TEST (aggregate per ordine) ──────────────
        y_pred_log = pipe.predict(X_test)
        if USE_LOG_TARGET:
            y_pred = _safe_expm1(y_pred_log)
            y_test_eval = y_test_raw
        else:
            y_pred = y_pred_log
            y_test_eval = y_test
        ids_agg, y_te_agg, yp_te_agg = aggregate_predictions_by_order(id_test, y_test_eval, y_pred)
        mae  = mean_absolute_error(y_te_agg, yp_te_agg)
        mape = mean_absolute_percentage_error(y_te_agg, yp_te_agg) * 100
        _log(f"  [TEST]  MAE={mae:.2f}, MAPE={mape:.2f}% (aggregato per ordine, {len(ids_agg)} ordini)")
        _log(f"  [GAP]   Overfit gap: {mape - mape_train:.2f}%")

        # Metriche per-riga (per diagnostica)
        mape_row = mean_absolute_percentage_error(y_test_eval, y_pred) * 100
        _log(f"  [TEST]  MAPE per-riga={mape_row:.2f}% (non aggregato)")

        # ── Incertezza ────────────────────────────────────────
        preproc = pipe.named_steps["prep"]
        fitted_model = pipe.named_steps["model"]
        X_test_tx = preproc.transform(X_test)
        pred_transform = _safe_expm1 if USE_LOG_TARGET else None
        mean_pred, std_pred, intervals = per_prediction_uncertainty(
            fitted_model, X_test_tx, INTERVAL_PERCENT, model_name, pred_transform=pred_transform
        )

        # GMM
        gmm = gmm_fit_on_target(y_train_raw, max_components=MAX_GMM_COMPONENTS, random_state=RANDOM_STATE)
        scale_global = gmm_mixture_std(gmm)
        confidence = confidence_from_scale(std_pred, scale_global)

        # Output DataFrame (per-riga)
        eps = 1e-9
        ape_percent = np.abs((y_test_eval - mean_pred) / (np.abs(y_test_eval) + eps)) * 100.0

        out_df = pd.DataFrame({
            ID_COL: id_test,
            "y_true": y_test_eval,
            "y_pred": mean_pred,
            "pred_std": std_pred,
            "conf_approx": confidence,
            f"pi{INTERVAL_PERCENT}_lower": intervals[:, 0],
            f"pi{INTERVAL_PERCENT}_upper": intervals[:, 1],
            "ape_percent": ape_percent,
        })

        predictions_per_tipo[tipo] = out_df
        results_summary[tipo] = {
            "model": model_name,
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
            "n_train_orders": len(ordini_train),
            "n_test_orders": len(ordini_test),
            "MAE_train": round(mae_train, 2),
            "MAPE_train_%": round(mape_train, 2),
            "MAE_test": round(mae, 2),
            "MAPE_test_%": round(mape, 2),
            "MAPE_test_row_%": round(mape_row, 2),
            "Overfit_gap_%": round(mape - mape_train, 2),
        }

        # Metadata per salvataggio
        if model_name == "LightGBM":
            model_params = model_params_used or {k: v for k, v in LGBM_PARAMS.items() if k != "verbose"}
        elif model_name == "XGBoost":
            model_params = dict(XGB_PARAMS)
        else:
            model_params = {}

        meta = {
            "tipo": tipo,
            "model_type": model_name,
            "model_params": model_params,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "split_method": "GroupShuffleSplit (per ordine)",
            "interval_percent": INTERVAL_PERCENT,
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
            "n_train_orders": len(ordini_train),
            "n_test_orders": len(ordini_test),
            "scale_method": SCALE_METHOD,
            "gmm_components": int(gmm.n_components),
            "target_transform": "log1p" if USE_LOG_TARGET else "none",
            "feature_selection": FEATURE_SELECTION,
            "optuna_tuning": USE_OPTUNA and model_name == "LightGBM",
            "MAE_test": round(mae, 2),
            "MAPE_test": round(mape, 2),
            "Overfit_gap": round(mape - mape_train, 2),
        }
        save_model_bundle(tipo, pipe, feature_cols, meta, gmm=gmm)

        # ── Plots (su dati aggregati per ordine) ──────────────
        plt.figure()
        plt.scatter(y_te_agg, yp_te_agg, alpha=0.4, s=10)
        lims = [min(y_te_agg.min(), yp_te_agg.min()), max(y_te_agg.max(), yp_te_agg.max())]
        plt.plot(lims, lims, "r--", alpha=0.5)
        plt.xlabel("y_true (ordine)")
        plt.ylabel("y_pred (ordine)")
        plt.title(f"True vs Pred — {tipo} [{model_name}] (MAPE={mape:.1f}%)")
        _savefig(f"scatter_true_vs_pred_{tipo}.png")

        plt.figure()
        plt.hist(y_te_agg - yp_te_agg, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel("Residuo (y_true - y_pred)")
        plt.ylabel("Frequenza")
        plt.title(f"Residui — {tipo} [{model_name}]")
        _savefig(f"residuals_{tipo}.png")

        # APE distribution (aggregato)
        eps_agg = 1e-9
        ape_agg = np.abs((y_te_agg - yp_te_agg) / (np.abs(y_te_agg) + eps_agg)) * 100.0
        vals = ape_agg[(ape_agg >= 0) & (ape_agg <= 100)]
        if vals.size > 0:
            plt.figure()
            plt.hist(vals, bins=np.arange(0, 101, 1), edgecolor="black", alpha=0.7)
            plt.xlabel("APE (%)")
            plt.ylabel("Frequenza")
            plt.title(f"Distribuzione APE% — {tipo} [{model_name}]")
            _savefig(f"ape_distribution_{tipo}.png")

    return results_summary, predictions_per_tipo


# ── Main ─────────────────────────────────────────────────

def main():
    input_file  = "04_matchato.xlsx"
    output_file = "06_training_results.xlsx"

    _log(f"[LOAD] Caricamento da {input_file}...")
    df_raw = load_matched_data(input_file)
    _log(f"  Righe totali: {len(df_raw)}")
    _log(f"  Ordini unici: {df_raw[ID_COL].nunique()}")
    _log(f"  Media quotazioni/ordine: {len(df_raw) / max(df_raw[ID_COL].nunique(), 1):.1f}")

    # NO collapse — lavoriamo su tutte le quotazioni con split per ordine
    df = df_raw.copy()

    # Drop colonne intermedie
    drop_cols = ["importo", "importotrasp", "stato_quotazione", "ordine_originale"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Modelli usati
    _log(f"\n[CONFIG] Modello per tipo_carico:")
    for tipo, model_name in MODEL_PER_TIPO.items():
        _log(f"  {tipo}: {model_name}")
    _log(f"[CONFIG] Split: GroupShuffleSplit per ordine (test_size={TEST_SIZE})")

    # Training
    summary, preds = fit_and_evaluate_by_tipo(df)

    # Export
    excel_sheets = {}

    # Dati completi
    excel_sheets["AllRows"] = df.copy()

    # Test per tipo (peggiori 2000 per APE)
    for tipo in ["Completo", "Parziale", "Groupage"]:
        if tipo in preds and not preds[tipo].empty:
            worst = preds[tipo].sort_values("ape_percent", ascending=False).head(2000)
            excel_sheets[f"Test_{tipo}"[:31]] = worst

    # Summary
    if summary:
        summary_df = pd.DataFrame([
            {"tipo_carico": tipo, **metrics}
            for tipo, metrics in summary.items()
        ])
        excel_sheets["Summary"] = summary_df
        _log(f"\n{'='*60}")
        _log("RIEPILOGO TRAINING")
        _log(f"{'='*60}")
        _log(summary_df.to_string(index=False))

    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        for sheet_name, df_sheet in excel_sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    _log(f"\n[OK] Risultati salvati in {output_file}")
    _log(f"[OK] Modelli salvati in {MODELDIR}/")


if __name__ == "__main__":
    main()
