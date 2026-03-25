# -*- coding: utf-8 -*-
"""
Jeden plik pod Kaggle Churn — maksymalizacja ROC-AUC (prawdopodobieństwa).

W Colab: wgraj train.csv i test.csv do /content (obok notebooka / ten sam katalog),
  albo uruchom: os.chdir("/content") jeśli pliki tam leżą.
Nie skanuje dysku — oczekuje dokładnie: train.csv, test.csv w bieżącym katalogu.

Opcjonalnie pierwsza linia:
  # !pip -q install pandas numpy scikit-learn lightgbm xgboost catboost

Tryb czasu / jakości (zmienna środowiskowa):
  CHURN_PRESET=balanced (domyślnie) | fast | max
  - fast: ~3–4× krócej niż max, nieco inna wariancja OOF
  - max: 10 foldów × 5 seedów (~jak wcześniej, na noc / final submit)
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMIT_FILE = "submission.csv"

TARGET_COL = "Churn"
ID_COL = "id"

MAX_CAT_CARD = 64
DROP_UNIQUES_GE = 800


def _training_profile() -> tuple[int, tuple[int, ...], int, int, int]:
    """N_SPLITS, SEEDS, EARLY_STOP, MAX_TREES, HGB_MAX_ITER"""
    p = (os.environ.get("CHURN_PRESET") or "balanced").lower()
    if p == "max":
        return 10, (42, 7, 99, 2026, 314), 200, 8000, 400
    if p == "fast":
        return 7, (42, 1337, 2026), 100, 5000, 220
    # balanced: dobry kompromis czas vs stabilność na A100
    return 8, (42, 1337, 2026), 140, 6500, 300


N_SPLITS, SEEDS, EARLY_STOP, MAX_TREES, HGB_MAX_ITER = _training_profile()

def _pip(*pkgs: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


for _mod, _pipn in (
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
):
    if importlib.util.find_spec(_mod.split(".")[0]) is None:
        _pip(_pipn)

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:
    TargetEncoder = None

import lightgbm as lgb
from xgboost import XGBClassifier


def guess_gpu() -> bool:
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


USE_GPU = guess_gpu()


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isfile(TRAIN_FILE):
        raise FileNotFoundError(
            f"Brak {TRAIN_FILE!r} w katalogu {os.getcwd()!r}. "
            "W Colab wgraj train.csv i test.csv do /content (Files → Upload)."
        )
    if not os.path.isfile(TEST_FILE):
        raise FileNotFoundError(
            f"Brak {TEST_FILE!r} w katalogu {os.getcwd()!r}."
        )
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    return train, test


def coerce_y(s: pd.Series) -> np.ndarray:
    if s.dtype == object or str(s.dtype) == "string":
        m = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        low = s.astype(str).str.strip().str.lower()
        if low.isin(m.keys()).mean() > 0.5:
            mapped = low.map(m)
            mapped = mapped.fillna(0).astype(np.float64).round().clip(0, 1)
            return mapped.astype(np.int32).to_numpy()

    v = pd.to_numeric(s, errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan)
    if v.notna().mean() > 0.99:
        finite = v.dropna().to_numpy(dtype=np.float64)
        if finite.size:
            u = np.unique(np.rint(finite).astype(np.int64))
            if len(u) <= 2 and set(u).issubset({0, 1}):
                return (
                    v.fillna(0)
                    .astype(np.float64)
                    .round()
                    .clip(0, 1)
                    .astype(np.int32)
                    .to_numpy()
                )

    from sklearn.preprocessing import LabelEncoder

    le_arr = LabelEncoder().fit_transform(s.astype(str))
    return le_arr.astype(np.int32)


@dataclass
class Plan:
    num_cols: list[str]
    cat_cols: list[str]
    drop_cols: list[str]


def plan_features(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> Plan:
    drop: list[str] = []
    num: list[str] = []
    cat: list[str] = []
    skip = {ID_COL, TARGET_COL}
    for c in X_tr.columns:
        if c in skip:
            continue
        if X_tr[c].dtype == object:
            n1, n2 = X_tr[c].nunique(dropna=False), X_te[c].nunique(dropna=False) if c in X_te.columns else n1
            if max(n1, n2) >= DROP_UNIQUES_GE:
                drop.append(c)
            else:
                cat.append(c)
            continue
        if pd.api.types.is_bool_dtype(X_tr[c]):
            cat.append(c)
            continue
        if pd.api.types.is_integer_dtype(X_tr[c]) and X_tr[c].nunique() <= MAX_CAT_CARD:
            cat.append(c)
            continue
        num.append(c)
    return Plan(num_cols=num, cat_cols=cat, drop_cols=drop)


def add_fe(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    o = df.copy()
    if not num_cols:
        return o
    Xn = o[num_cols].replace([np.inf, -np.inf], np.nan).astype(np.float64)
    o["__na_cnt"] = Xn.isna().sum(axis=1).astype(np.float32)
    o["__row_med"] = Xn.median(axis=1, skipna=True).astype(np.float32)
    for c in num_cols[:8]:
        v = pd.to_numeric(o[c], errors="coerce")
        if v.notna().all() and v.min() >= 0 and v.max() > 30:
            o[f"__log1p_{c}"] = np.log1p(v).astype(np.float32)
    return o


def refe_plan(X: pd.DataFrame) -> Plan:
    num, cat = [], []
    for c in X.columns:
        if c == ID_COL:
            continue
        if X[c].dtype == object or str(X[c].dtype) == "category" or pd.api.types.is_bool_dtype(X[c]):
            cat.append(c)
        elif pd.api.types.is_integer_dtype(X[c]) and X[c].nunique(dropna=False) <= MAX_CAT_CARD:
            cat.append(c)
        else:
            num.append(c)
    return Plan(num_cols=num, cat_cols=cat, drop_cols=[])


def build_sklearn_prep(p: Plan, seed: int) -> ColumnTransformer:
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    parts: list = []
    if p.num_cols:
        parts.append(("n", num_pipe, p.num_cols))
    if p.cat_cols:
        if TargetEncoder is not None:
            te = TargetEncoder(random_state=seed, cv=min(5, N_SPLITS), shuffle=True)
            parts.append(("c", te, p.cat_cols))
        else:
            cat_pipe = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            )
            parts.append(("c", cat_pipe, p.cat_cols))
    if not parts:
        raise ValueError("Brak cech po preprocessingu")
    return ColumnTransformer(transformers=parts, remainder="drop", sparse_threshold=0.0)


def catboost_pool(X: pd.DataFrame, cat_names: list[str], y=None) -> Pool:
    d = X.copy()
    for c in cat_names:
        if c in d.columns:
            d[c] = d[c].astype(str).replace({"nan": "missing"}).fillna("missing")
    idx = [d.columns.get_loc(c) for c in cat_names if c in d.columns]
    if y is None:
        return Pool(d, cat_features=idx)
    return Pool(d, y, cat_features=idx)


def lgb_cat_frame(X: pd.DataFrame, cats: list[str], ref: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    o = X.copy()
    feat: list[str] = []
    for c in cats:
        if c not in o.columns:
            continue
        ref_s = ref[c].astype(str)
        cur = o[c].astype(str)
        u = pd.unique(pd.concat([ref_s, cur], ignore_index=True))
        o[c] = pd.Categorical(cur, categories=u, ordered=False)
        feat.append(c)
    for c in o.columns:
        if c not in feat and o[c].dtype == object:
            o[c] = pd.to_numeric(o[c], errors="coerce")
    return o, feat


def main() -> None:
    train, test = load_frames()
    if TARGET_COL not in train.columns:
        raise ValueError(f"Kolumna {TARGET_COL} musi być w train.csv")
    if ID_COL not in train.columns or ID_COL not in test.columns:
        raise ValueError(f"Wymagana kolumna {ID_COL}")

    y = coerce_y(train[TARGET_COL])
    print(f"CHURN_PRESET={os.environ.get('CHURN_PRESET', 'balanced')!r} | GPU={USE_GPU} | folds={N_SPLITS} seeds={len(SEEDS)}")

    raw_plan = plan_features(train.drop(columns=[TARGET_COL]), test)
    dropc = raw_plan.drop_cols
    tr_x = train.drop(columns=[TARGET_COL] + dropc, errors="ignore").copy()
    te_x = test.drop(columns=dropc, errors="ignore").copy()
    ids = test[ID_COL].values

    tr_x = add_fe(tr_x, raw_plan.num_cols)
    te_x = add_fe(te_x, raw_plan.num_cols)
    tr_x = tr_x.drop(columns=[ID_COL], errors="ignore")
    te_x = te_x.drop(columns=[ID_COL], errors="ignore")
    plan = refe_plan(tr_x)

    n = len(tr_x)
    n_test = len(te_x)

    oof_xgb = np.zeros(n)
    oof_lgb = np.zeros(n)
    oof_cat = np.zeros(n)
    oof_hgb = np.zeros(n)

    pred_xgb = np.zeros(n_test)
    pred_lgb = np.zeros(n_test)
    pred_cat = np.zeros(n_test)
    pred_hgb = np.zeros(n_test)

    n_bags = len(SEEDS)

    total_steps = len(SEEDS) * N_SPLITS
    step = 0
    start_wall = time.time()
    last_print_wall = 0.0

    for si, seed in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr_i, va_i) in enumerate(skf.split(tr_x, y)):
            X_tr, X_va = tr_x.iloc[tr_i], tr_x.iloc[va_i]
            y_tr, y_va = y[tr_i], y[va_i]

            pre = build_sklearn_prep(plan, seed + fold)
            pre.fit(X_tr, y_tr)
            Xt = pre.transform(X_tr)
            Xv = pre.transform(X_va)
            Xe = pre.transform(te_x)
            if isinstance(Xt, np.ndarray):
                Xt = Xt.astype(np.float32, copy=False)
            if isinstance(Xv, np.ndarray):
                Xv = Xv.astype(np.float32, copy=False)
            if isinstance(Xe, np.ndarray):
                Xe = Xe.astype(np.float32, copy=False)

            xgb_kw = dict(
                n_estimators=MAX_TREES,
                learning_rate=0.02,
                max_depth=7,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.72,
                reg_alpha=0.08,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                random_state=seed + fold,
                n_jobs=-1,
            )
            if USE_GPU:
                xgb_kw["device"] = "cuda"
                xgb_kw["predictor"] = "gpu_predictor"
                xgb_kw["max_bin"] = 256
            mx = XGBClassifier(**xgb_kw)
            try:
                mx.fit(Xt, y_tr, eval_set=[(Xv, y_va)], verbose=False, early_stopping_rounds=EARLY_STOP)
            except TypeError:
                mx.fit(Xt, y_tr, eval_set=[(Xv, y_va)], verbose=False)

            pv = mx.predict_proba(Xv)[:, 1]
            pe = mx.predict_proba(Xe)[:, 1]
            oof_xgb[va_i] += pv / n_bags
            pred_xgb += pe / (n_bags * N_SPLITS)

            X_tr_l, cf = lgb_cat_frame(X_tr, plan.cat_cols, X_tr)
            X_va_l, _ = lgb_cat_frame(X_va, plan.cat_cols, X_tr)
            X_te_l, _ = lgb_cat_frame(te_x, plan.cat_cols, X_tr)
            dtr = lgb.Dataset(X_tr_l, label=y_tr, categorical_feature=cf, free_raw_data=False)
            dva = lgb.Dataset(X_va_l, label=y_va, categorical_feature=cf, reference=dtr, free_raw_data=False)
            bp = dict(
                objective="binary",
                metric="auc",
                learning_rate=0.03,
                num_leaves=63,
                min_data_in_leaf=35,
                feature_fraction=0.75,
                bagging_fraction=0.8,
                bagging_freq=1,
                lambda_l1=0.1,
                lambda_l2=2.0,
                verbosity=-1,
                seed=seed + fold,
                n_jobs=-1,
            )
            if USE_GPU:
                bp.update(
                    {
                        "device_type": "gpu",
                        "gpu_platform_id": 0,
                        "gpu_device_id": 0,
                        "max_bin": 255,
                    }
                )
            try:
                bst = lgb.train(
                    bp,
                    dtr,
                    num_boost_round=MAX_TREES,
                    valid_sets=[dva],
                    callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
                )
            except Exception:
                if USE_GPU:
                    bp["device_type"] = "cpu"
                    bst = lgb.train(
                        bp,
                        dtr,
                        num_boost_round=MAX_TREES,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
                    )
                else:
                    raise
            pv = bst.predict(X_va_l, num_iteration=bst.best_iteration)
            pe = bst.predict(X_te_l, num_iteration=bst.best_iteration)
            oof_lgb[va_i] += pv / n_bags
            pred_lgb += pe / (n_bags * N_SPLITS)

            train_pool = catboost_pool(X_tr, plan.cat_cols, y_tr)
            val_pool = catboost_pool(X_va, plan.cat_cols, y_va)
            te_pool = catboost_pool(te_x, plan.cat_cols, None)
            _cat_eval = "Logloss" if USE_GPU else "AUC"
            ckw = dict(
                iterations=MAX_TREES,
                learning_rate=0.03,
                depth=8,
                l2_leaf_reg=8.0,
                random_strength=0.75,
                loss_function="Logloss",
                eval_metric=_cat_eval,
                random_seed=seed + fold,
                od_type="Iter",
                od_wait=EARLY_STOP,
                verbose=False,
                allow_writing_files=False,
                task_type="GPU" if USE_GPU else "CPU",
                thread_count=-1,
            )
            mc = CatBoostClassifier(**ckw)
            try:
                mc.fit(train_pool, eval_set=val_pool, use_best_model=True)
            except Exception:
                ckw["task_type"] = "CPU"
                ckw["eval_metric"] = "AUC"
                mc = CatBoostClassifier(**ckw)
                mc.fit(train_pool, eval_set=val_pool, use_best_model=True)
            pv = mc.predict_proba(val_pool)[:, 1]
            pe = mc.predict_proba(te_pool)[:, 1]
            oof_cat[va_i] += pv / n_bags
            pred_cat += pe / (n_bags * N_SPLITS)

            mh = HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_depth=10,
                max_iter=HGB_MAX_ITER,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=seed + fold,
                early_stopping=True,
                validation_fraction=0.08,
                n_iter_no_change=min(EARLY_STOP, 80),
            )
            mh.fit(Xt, y_tr)
            pv = mh.predict_proba(Xv)[:, 1]
            pe = mh.predict_proba(Xe)[:, 1]
            oof_hgb[va_i] += pv / n_bags
            pred_hgb += pe / (n_bags * N_SPLITS)

            if si == 0 and fold == 0:
                print(
                    f"Trening: GPU={USE_GPU} | {N_SPLITS} folds × {len(SEEDS)} seeds | "
                    "XGBoost + LightGBM + CatBoost + HistGradientBoosting → logit stacking + blend"
                )

            step += 1
            now = time.time()
            if (now - last_print_wall) > 0.1 or step == total_steps:
                elapsed = now - start_wall
                avg = elapsed / max(1, step)
                remaining = avg * (total_steps - step)
                pct = 100.0 * step / total_steps
                eta_m = int(remaining // 60)
                eta_s = int(remaining % 60)
                print(f"loading: {step}/{total_steps} ({pct:.1f}%) | ETA ~ {eta_m:d}m {eta_s:d}s")
                last_print_wall = now

    stack_tr = np.column_stack([oof_xgb, oof_lgb, oof_cat, oof_hgb])
    stack_te = np.column_stack([pred_xgb, pred_lgb, pred_cat, pred_hgb])

    meta = LogisticRegression(max_iter=2500, C=0.35, solver="lbfgs")
    meta.fit(stack_tr, y)
    oof_m = meta.predict_proba(stack_tr)[:, 1]
    te_m = meta.predict_proba(stack_te)[:, 1]

    def rank(v: np.ndarray) -> np.ndarray:
        r = pd.Series(v).rank(method="average").to_numpy(dtype=float)
        return (r - 1.0) / max(1.0, len(r) - 1.0)

    def rank_stack(*cols: np.ndarray) -> np.ndarray:
        return np.mean(np.column_stack([rank(c) for c in cols]), axis=1)

    oof_r_all = rank_stack(oof_xgb, oof_lgb, oof_cat, oof_hgb)
    te_r_all = rank_stack(pred_xgb, pred_lgb, pred_cat, pred_hgb)

    auc_x = roc_auc_score(y, oof_xgb)
    auc_l = roc_auc_score(y, oof_lgb)
    auc_c = roc_auc_score(y, oof_cat)
    auc_h = roc_auc_score(y, oof_hgb)
    w_raw = np.maximum(np.array([auc_x, auc_l, auc_c, auc_h]) - 0.5, 0.02) ** 2
    w_base = w_raw / w_raw.sum()
    oof_w = w_base[0] * oof_xgb + w_base[1] * oof_lgb + w_base[2] * oof_cat + w_base[3] * oof_hgb
    te_w = w_base[0] * pred_xgb + w_base[1] * pred_lgb + w_base[2] * pred_cat + w_base[3] * pred_hgb

    oof_lc = rank_stack(oof_lgb, oof_cat)
    te_lc = rank_stack(pred_lgb, pred_cat)

    best_oof = -1.0
    best_a, best_b = 0.55, 0.35
    for a in np.linspace(0.0, 1.0, 23):
        for b in np.linspace(0.0, 1.0, 23):
            if a + b > 1.0 + 1e-9:
                continue
            c = 1.0 - a - b
            o = a * oof_m + b * oof_w + c * oof_r_all
            sc = roc_auc_score(y, o)
            if sc > best_oof:
                best_oof = sc
                best_a, best_b = a, b

    best_c = max(0.0, 1.0 - best_a - best_b)
    blend_te = best_a * te_m + best_b * te_w + best_c * te_r_all
    blend_oof = best_a * oof_m + best_b * oof_w + best_c * oof_r_all

    legacy = 0.72 * oof_m + 0.28 * oof_r_all

    print("--- OOF ROC-AUC ---")
    print("XGB:", auc_x)
    print("LGB:", auc_l)
    print("CAT:", auc_c)
    print("HGB:", auc_h)
    print("STACK(meta):", roc_auc_score(y, oof_m))
    print("wagi_modeli (do blendu liniowego):", np.round(w_base, 4).tolist())
    print("blend_grid OOF:", best_oof, f"(a=meta {best_a:.3f}, b=lin_w {best_b:.3f}, c=rank {best_c:.3f})")
    print("blend stary (0.72meta+0.28rank):", roc_auc_score(y, legacy))
    print("BLEND_final (jak submission):", roc_auc_score(y, blend_oof))

    out = pd.DataFrame({ID_COL: ids, TARGET_COL: np.clip(blend_te, 1e-7, 1 - 1e-7)})
    out.to_csv(SUBMIT_FILE, index=False)
    print("Zapisano:", os.path.abspath(SUBMIT_FILE))

    pd.DataFrame({ID_COL: ids, TARGET_COL: np.clip(te_m, 1e-7, 1 - 1e-7)}).to_csv(
        SUBMIT_FILE.replace(".csv", "_stack_only.csv"), index=False
    )
    pd.DataFrame({ID_COL: ids, TARGET_COL: np.clip(te_lc, 1e-7, 1 - 1e-7)}).to_csv(
        SUBMIT_FILE.replace(".csv", "_rank_lgb_cat.csv"), index=False
    )


if __name__ == "__main__":
    main()
