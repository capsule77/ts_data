# MBS Quant Research — Data Analysis Cheatsheet
### Squarepoint Capital Interview Preparation

---

## Table of Contents
1. [General Project Procedure](#1-general-project-procedure)
2. [Data Loading & Quality](#2-data-loading--quality)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Feature Engineering](#4-feature-engineering)
5. [Train / Test Split](#5-train--test-split)
6. [Prediction Tasks & Targets](#6-prediction-tasks--targets)
7. [Model Setup & Hyperparameters](#7-model-setup--hyperparameters)
8. [Training & Cross-Validation](#8-training--cross-validation)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Residual Diagnostics](#10-residual-diagnostics)
11. [Feature Importance](#11-feature-importance)
12. [Forecasting (ARIMA / time series)](#12-forecasting-arima--time-series)
13. [Stationarity & Decomposition](#13-stationarity--decomposition)
14. [Common Pitfalls](#14-common-pitfalls)
15. [Interview Presentation Framework](#15-interview-presentation-framework)

---

## 1. General Project Procedure

```
Step 1  →  Inspect        load, shape, dtypes, nulls, date range
Step 2  →  EDA            distributions, time plots, correlation heatmap
Step 3  →  Feature Eng.   refi incentive, burnout, lags, seasonality
Step 4  →  Split          time-based (NEVER random) — 80/20 date cutoff
Step 5  →  Baseline       naïve model (CPR_t as T+1 forecast)
Step 6  →  Models         Ridge → RF → XGB → LGB → Stacking
Step 7  →  Evaluate       RMSE/R² (regression) · AUC/F1 (classification)
Step 8  →  Diagnose       residual ACF, calibration, permutation importance
```

---

## 2. Data Loading & Quality

```python
import numpy as np
import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('mbs_data.csv', parse_dates=['date'])
df = df.sort_values(['pool_id', 'date']).reset_index(drop=True)

# ── Shape & types ─────────────────────────────────────────────────────────────
print(df.shape)           # (rows, cols)
print(df.dtypes)
print(df.head())

# ── Missing values ────────────────────────────────────────────────────────────
df.isnull().sum()                              # per column
df.isnull().sum() / len(df)                    # fraction missing
df.isnull().any(axis=1).sum()                  # rows with any null

# ── Date range & coverage ─────────────────────────────────────────────────────
print(df['date'].min(), df['date'].max())
print(df['date'].nunique())                    # how many months
print(df.groupby('pool_id')['date'].count())   # observations per pool

# ── Cardinality ───────────────────────────────────────────────────────────────
df.nunique()
df['pool_id'].unique()

# ── Numeric summary ───────────────────────────────────────────────────────────
df.describe().round(2)
df.describe(percentiles=[.05, .25, .5, .75, .95]).round(2)

# ── Outlier detection ─────────────────────────────────────────────────────────
from scipy import stats
z_scores = np.abs(stats.zscore(df[['cpr', 'wac', 'ltv']].dropna()))
outlier_rows = (z_scores > 3).any(axis=1)
print(f'Outliers: {outlier_rows.sum()}')

# IQR method
Q1, Q3 = df['cpr'].quantile(0.25), df['cpr'].quantile(0.75)
IQR = Q3 - Q1
mask = (df['cpr'] < Q1 - 1.5*IQR) | (df['cpr'] > Q3 + 1.5*IQR)
print(df[mask][['date', 'pool_id', 'cpr']])
```

---

## 3. Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── CPR over time, all pools ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for pid in df['pool_id'].unique():
    sub = df[df['pool_id'] == pid]
    ax.plot(sub['date'], sub['cpr'], alpha=0.6, linewidth=1, label=pid)
ax.set_title('CPR Time Series — All Pools')
ax.set_ylabel('CPR (%)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()

# ── Distribution of CPR ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['cpr'].hist(bins=40, ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('CPR Distribution')
df.boxplot(column='cpr', by='pool_id', ax=axes[1], vert=False)
axes[1].set_title('CPR by Pool')
plt.tight_layout()

# ── Correlation heatmap ───────────────────────────────────────────────────────
numeric_cols = ['cpr', 'wac', 'wam', 'ltv', 'refi_index', 'rate_10y',
                'spread_to_treasury', 'factor']
corr = df[numeric_cols].corr().round(2)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.matshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha='left')
ax.set_yticklabels(numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, str(corr.iloc[i, j]), ha='center', va='center', fontsize=8)
plt.tight_layout()

# ── The prepayment S-curve ────────────────────────────────────────────────────
df['refi_incentive'] = df['wac'] - df['rate_10y']
df['refi_bin'] = pd.cut(df['refi_incentive'], bins=np.linspace(-2, 3.5, 25))
scurve = df.groupby('refi_bin')['cpr'].agg(['mean', 'std']).dropna()
x_mid = [interval.mid for interval in scurve.index]

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(x_mid, scurve['mean'] - scurve['std'],
                scurve['mean'] + scurve['std'], alpha=0.25, label='±1 std')
ax.plot(x_mid, scurve['mean'], 'o-', linewidth=2, label='Mean CPR')
ax.axvline(0, color='red', linestyle='--', label='At-the-money')
ax.set_xlabel('Refi Incentive: WAC − 10Y Rate (%)')
ax.set_ylabel('CPR (%)')
ax.set_title('Prepayment S-Curve')
ax.legend()

# ── Seasonality: CPR by month ─────────────────────────────────────────────────
monthly = df.groupby(df['date'].dt.month)['cpr'].mean()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(monthly.index, monthly.values, color=['#2ecc71' if v > monthly.mean() else '#e74c3c'
                                              for v in monthly.values])
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title('Average CPR by Month (seasonality)')

# ── Rolling correlation: refi incentive → CPR ─────────────────────────────────
from scipy.stats import spearmanr

pool = df[df['pool_id'] == 'POOL-001'].sort_values('date').set_index('date')
rolling_corr = pool['refi_incentive'].rolling(12).corr(pool['cpr'])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(rolling_corr.index, rolling_corr, color='darkorange', linewidth=2)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('12-Month Rolling Corr: Refi Incentive → CPR')
ax.set_ylabel('Spearman ρ')

rho, pval = spearmanr(df['refi_incentive'].dropna(), df['cpr'].dropna())
print(f'Spearman ρ: {rho:.3f}, p={pval:.2e}')
```

---

## 4. Feature Engineering

```python
def engineer_features(df):
    d = df.copy()

    # ── Refi signal (most important) ──────────────────────────────────────────
    d['refi_incentive']    = d['wac'] - d['rate_10y']        # in-the-money if >0
    d['refi_incentive_sq'] = d['refi_incentive'] ** 2        # S-curve curvature
    d['refi_incentive_cb'] = d['refi_incentive'] ** 3        # higher-order tail
    d['in_the_money']      = (d['refi_incentive'] > 0.5).astype(int)

    # ── Pool characteristics ───────────────────────────────────────────────────
    d['burnout']      = 1 - d['factor']                      # fraction already prepaid
    d['ltv_kink']     = np.clip(d['ltv'] - 80, 0, None)     # kink at 80% LTV threshold
    d['fico_norm']    = (d['fico'] - 720) / 50               # standardise around prime

    # ── Interaction terms ─────────────────────────────────────────────────────
    d['refi_x_burnout'] = d['refi_incentive'] * d['burnout'] # incentive dampened by burnout
    d['ltv_x_rate']     = d['ltv_kink'] * d['rate_10y']     # LTV sensitivity to rate

    # ── Seasonality (circular encoding — preserves Jan/Dec continuity) ─────────
    d['month_sin'] = np.sin(2 * np.pi * d['date'].dt.month / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['date'].dt.month / 12)
    d['quarter']   = d['date'].dt.quarter

    # ── Market context ────────────────────────────────────────────────────────
    d['spread_norm'] = d['spread_to_treasury'] / 100
    d['vix_norm']    = d['vix'] / 30

    # ── Lag features — MUST be computed per pool (never on full df!) ───────────
    for pool_id, grp in d.groupby('pool_id'):
        idx = grp.index
        d.loc[idx, 'cpr_lag1']    = grp['cpr'].shift(1)
        d.loc[idx, 'cpr_lag3']    = grp['cpr'].shift(3)
        d.loc[idx, 'cpr_lag6']    = grp['cpr'].shift(6)
        d.loc[idx, 'cpr_ma3']     = grp['cpr'].shift(1).rolling(3).mean()
        d.loc[idx, 'cpr_ma6']     = grp['cpr'].shift(1).rolling(6).mean()
        d.loc[idx, 'cpr_std3']    = grp['cpr'].shift(1).rolling(3).std()   # volatility
        d.loc[idx, 'rate_chg1']   = grp['rate_10y'].diff(1)
        d.loc[idx, 'rate_chg3']   = grp['rate_10y'].diff(3)
        d.loc[idx, 'refi_chg1']   = grp['refi_incentive'].diff(1)

    # ── Target variables ──────────────────────────────────────────────────────
    d['cpr_next']       = d.groupby('pool_id')['cpr'].shift(-1)        # T+1 CPR
    d['high_cpr']       = (d['cpr_next'] > d['cpr_next'].quantile(0.70)).astype(int)
    d['cpr_direction']  = (d.groupby('pool_id')['cpr'].shift(-1) > d['cpr']).astype(int)

    return d.dropna().reset_index(drop=True)


df = engineer_features(df_raw)

FEATURES = [
    # Core refi signal
    'refi_incentive', 'refi_incentive_sq', 'refi_incentive_cb', 'in_the_money',
    # Pool characteristics
    'burnout', 'ltv_kink', 'fico_norm', 'wam',
    # Seasonality
    'month_sin', 'month_cos',
    # Market
    'refi_index', 'refi_x_burnout', 'spread_norm', 'vix_norm',
    # Lags
    'cpr_lag1', 'cpr_lag3', 'cpr_ma3', 'cpr_std3', 'rate_chg1', 'rate_chg3',
]

print(f'Features: {len(FEATURES)}')
print(f'After dropna: {df.shape}')
```

---

## 5. Train / Test Split

```python
# ── ALWAYS split by time — never by row index ─────────────────────────────────
CUTOFF = '2023-01-01'
train = df[df['date'] <  CUTOFF].copy()
test  = df[df['date'] >= CUTOFF].copy()

X_train, X_test = train[FEATURES], test[FEATURES]
y_train_reg = train['cpr_next']
y_test_reg  = test['cpr_next']
y_train_clf = train['high_cpr']
y_test_clf  = test['high_cpr']

print(f'Train: {len(train):,} rows  {train.date.min().date()} → {train.date.max().date()}')
print(f'Test : {len(test):,} rows   {test.date.min().date()} → {test.date.max().date()}')

# ── Time-series cross-validation (for model selection) ───────────────────────
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

# Visualise the folds
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f'Fold {fold+1}: train={len(tr_idx)}, val={len(val_idx)}  '
          f'val dates: {train.iloc[val_idx]["date"].min().date()} → '
          f'{train.iloc[val_idx]["date"].max().date()}')

# ── Naïve baseline ────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

naive_pred_reg = test['cpr'].values                    # use current CPR as T+1 forecast
naive_rmse     = np.sqrt(mean_squared_error(y_test_reg, naive_pred_reg))
naive_acc      = max(y_test_clf.mean(), 1 - y_test_clf.mean())  # always predict majority
print(f'Naïve RMSE: {naive_rmse:.3f}')
print(f'Naïve Accuracy: {naive_acc:.3f}')
```

---

## 6. Prediction Tasks & Targets

| Task | Target column | Type | Threshold | Metric |
|---|---|---|---|---|
| CPR Regression | `cpr_next` | Continuous | — | RMSE, R² |
| High-CPR Flag | `high_cpr` | Binary (0/1) | 70th percentile | AUC-ROC, F1 |
| Direction Change | `cpr_direction` | Binary (0/1) | CPR rising vs falling | AUC-ROC, Accuracy |

```python
# Task 1: regression target
y_reg = df.groupby('pool_id')['cpr'].shift(-1)   # next month's CPR

# Task 2: classification — top-30% prepayment pools
threshold = df['cpr_next'].quantile(0.70)
y_clf     = (df['cpr_next'] > threshold).astype(int)
print(f'High-CPR threshold: {threshold:.1f}%  |  positive rate: {y_clf.mean():.2%}')

# Task 3: direction — will CPR rise or fall next month?
y_dir = (df.groupby('pool_id')['cpr'].shift(-1) > df['cpr']).astype(int)
print(f'Rising CPR rate: {y_dir.mean():.2%}  (near 50% → hard task)')
```

---

## 7. Model Setup & Hyperparameters

```python
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               StackingRegressor, StackingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# ── REGRESSION MODELS ─────────────────────────────────────────────────────────

# Ridge: L2 regularisation. Scale first — alpha is sensitive to feature scale.
ridge_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=10.0))          # alpha: try [0.1, 1, 10, 100] via CV
])

# Lasso: L1 regularisation → automatic feature selection (some coefs → 0)
lasso_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Lasso(alpha=0.1,            # smaller alpha = less sparse
                     max_iter=5000,        # increase if "did not converge"
                     selection='cyclic'))
])

# Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=200,          # more trees = lower variance; 200 usually enough
    max_depth=8,               # cap depth to prevent overfitting on time series
    min_samples_leaf=5,        # minimum 5 samples per leaf — smooths predictions
    max_features='sqrt',       # default; 'log2' for more randomness
    n_jobs=-1,                 # use all CPU cores
    random_state=42
)

# XGBoost Regressor
xgb_reg = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,        # low lr + more trees → better generalisation
    max_depth=5,               # shallower = more regularised
    subsample=0.8,             # row subsampling per tree
    colsample_bytree=0.8,      # feature subsampling per tree
    reg_alpha=0.1,             # L1 regularisation on leaf weights
    reg_lambda=1.0,            # L2 regularisation on leaf weights
    min_child_weight=5,        # minimum sum of instance weight in leaf
    random_state=42,
    verbosity=0                # suppress output
)

# LightGBM Regressor
lgb_reg = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,             # must be < 2^max_depth; controls model complexity
    subsample=0.8,
    colsample_bytree=0.8,      # called feature_fraction in native API
    reg_alpha=0.1,
    min_child_samples=20,      # min data in leaf — critical for avoiding overfit
    random_state=42,
    verbose=-1                 # suppress all output
)

# Stacking Ensemble — base models feed meta-learner
stack_reg = StackingRegressor(
    estimators=[
        ('ridge', Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=10.0))])),
        ('xgb',   xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                    max_depth=4, random_state=42, verbosity=0)),
        ('lgb',   lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                     max_depth=5, random_state=42, verbose=-1)),
    ],
    final_estimator=Ridge(alpha=5.0),      # meta-learner on out-of-fold predictions
    cv=5,                                  # use int, NOT TimeSeriesSplit (raises error)
    passthrough=False                      # don't pass raw X to meta-learner
)


# ── CLASSIFICATION MODELS ─────────────────────────────────────────────────────

# Logistic Regression (L2)
lr_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(
        C=0.5,                 # inverse of regularisation strength (smaller = stronger)
        penalty='l2',
        solver='lbfgs',        # good default for L2
        max_iter=1000,
        random_state=42
    ))
])

# Logistic Regression (L1 — sparse)
lr_l1_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(
        C=0.5,
        penalty='l1',
        solver='saga',         # saga supports L1
        max_iter=2000,
        random_state=42
    ))
])

# Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight='balanced',   # upweights minority class automatically
    n_jobs=-1,
    random_state=42
)

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2,        # = neg_count/pos_count for imbalanced targets
    eval_metric='logloss',     # suppress deprecation warning
    random_state=42,
    verbosity=0
)

# LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

# Stacking Classifier
stack_clf = StackingClassifier(
    estimators=[
        ('lr',  Pipeline([('sc', StandardScaler()),
                          ('m',  LogisticRegression(C=0.5, max_iter=1000))])),
        ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=4, eval_metric='logloss',
                                    random_state=42, verbosity=0)),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=5, random_state=42, verbose=-1)),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=500),
    cv=5,
    stack_method='predict_proba'    # use probability outputs for meta-learner
)
```

---

## 8. Training & Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# ── Fit a single model ────────────────────────────────────────────────────────
model.fit(X_train, y_train_reg)
y_pred = model.predict(X_test)

# ── Time-series cross-validation (on train set) ───────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

# Single metric
cv_scores = cross_val_score(
    model, X_train, y_train_reg,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
print(f'CV RMSE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

# Multiple metrics at once
cv_results = cross_validate(
    model, X_train, y_train_reg,
    cv=tscv,
    scoring={
        'rmse': make_scorer(mean_squared_error, squared=False, greater_is_better=False),
        'r2':   'r2'
    },
    return_train_score=True,
    n_jobs=-1
)
print(pd.DataFrame(cv_results).mean().round(3))

# ── Manual fold loop (more control, inspect per-fold behaviour) ───────────────
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_reg.iloc[tr_idx], y_train_reg.iloc[val_idx]

    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    rmse  = np.sqrt(mean_squared_error(y_val, preds))
    r2    = r2_score(y_val, preds)
    print(f'Fold {fold+1}: RMSE={rmse:.3f}  R²={r2:.3f}')

# ── Hyperparameter search with time-series CV ─────────────────────────────────
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'model__alpha': [0.1, 1.0, 10.0, 100.0]
}
gs = GridSearchCV(
    ridge_reg,
    param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    refit=True
)
gs.fit(X_train, y_train_reg)
print(f'Best alpha: {gs.best_params_}  |  Best RMSE: {-gs.best_score_:.3f}')

# XGBoost / LightGBM with early stopping
xgb_reg.fit(
    X_train, y_train_reg,
    eval_set=[(X_test, y_test_reg)],
    verbose=False
)
# Note: early_stopping_rounds was moved to constructor in XGB 2.x:
xgb_reg_es = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    early_stopping_rounds=30,   # stop if no improvement for 30 rounds
    random_state=42,
    verbosity=0
)
xgb_reg_es.fit(X_train, y_train_reg,
               eval_set=[(X_test, y_test_reg)],
               verbose=False)
print(f'Best iteration: {xgb_reg_es.best_iteration}')
```

---

## 9. Evaluation Metrics

### Regression

```python
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, mean_absolute_percentage_error)

y_pred = model.predict(X_test)

rmse  = np.sqrt(mean_squared_error(y_test_reg, y_pred))
mae   = mean_absolute_error(y_test_reg, y_pred)
r2    = r2_score(y_test_reg, y_pred)
mape  = mean_absolute_percentage_error(y_test_reg, y_pred)  # % error

print(f'RMSE: {rmse:.3f} CPR pts')
print(f'MAE:  {mae:.3f} CPR pts')
print(f'R²:   {r2:.3f}')
print(f'MAPE: {mape:.2%}')
print(f'Naïve RMSE: {naive_rmse:.3f}  |  Skill: {(1-rmse/naive_rmse)*100:.1f}% improvement')

# Actual vs predicted plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test_reg, y_pred, alpha=0.3, s=10)
lims = [min(y_test_reg.min(), y_pred.min()), max(y_test_reg.max(), y_pred.max())]
ax.plot(lims, lims, 'r--')
ax.set_xlabel('Actual CPR'); ax.set_ylabel('Predicted CPR')
ax.set_title(f'Actual vs Predicted  R²={r2:.3f}')

# Time-series forecast plot (for one pool)
pool_mask = test['pool_id'] == 'POOL-001'
pool_test = test[pool_mask]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(pool_test['date'], y_test_reg[pool_mask], 'k-', label='Actual', linewidth=2)
ax.plot(pool_test['date'], y_pred[pool_mask.values], '--', label='Predicted', linewidth=1.5)
ax.legend(); ax.set_title('CPR Forecast — POOL-001')
```

### Classification

```python
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, classification_report,
                              roc_curve, precision_recall_curve,
                              average_precision_score)

y_prob = model.predict_proba(X_test)[:, 1]   # probability of positive class
y_pred = model.predict(X_test)

# Core metrics
acc  = accuracy_score(y_test_clf, y_pred)
auc  = roc_auc_score(y_test_clf, y_prob)
f1   = f1_score(y_test_clf, y_pred)
prec = precision_score(y_test_clf, y_pred)
rec  = recall_score(y_test_clf, y_pred)
ap   = average_precision_score(y_test_clf, y_prob)   # area under PR curve

print(f'Accuracy:  {acc:.3f}')
print(f'AUC-ROC:   {auc:.3f}')
print(f'F1:        {f1:.3f}')
print(f'Precision: {prec:.3f}  Recall: {rec:.3f}')
print(f'Avg Prec:  {ap:.3f}')

# Full report
print(classification_report(y_test_clf, y_pred, target_names=['Normal', 'High CPR']))

# Confusion matrix
cm = confusion_matrix(y_test_clf, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                color='white' if cm[i, j] > cm.max() * 0.5 else 'black')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal', 'High CPR'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Normal', 'High CPR'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test_clf, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve'); ax.legend()

# Precision-Recall curve (better for imbalanced classes)
prec_vals, rec_vals, _ = precision_recall_curve(y_test_clf, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_vals, prec_vals, linewidth=2)
ax.axhline(y_test_clf.mean(), color='red', linestyle='--', label='Random baseline')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall Curve  AP={ap:.3f}')
ax.legend()

# Threshold selection (optimise F1)
f1_scores = 2 * (prec_vals * rec_vals) / (prec_vals + rec_vals + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores[:-1])]
print(f'Optimal threshold: {best_threshold:.3f}')
y_pred_custom = (y_prob >= best_threshold).astype(int)
```

---

## 10. Residual Diagnostics

```python
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import statsmodels.api as sm

# ── Compute residuals ─────────────────────────────────────────────────────────
residuals = y_test_reg.values - y_pred

# ── 1. Residuals over time ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(test['date'], residuals, alpha=0.3, s=8)
ax.axhline(0, color='red', linestyle='--')
ax.set_title('Residuals Over Time')

# ── 2. Residuals vs Fitted ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_pred, residuals, alpha=0.3, s=8)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Fitted values'); ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Fitted  (fan shape → heteroskedasticity)')

# ── 3. QQ plot (normality check) ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
stats.probplot(residuals, dist='norm', plot=ax)
ax.set_title('Normal Q-Q Plot')

_, p_shapiro = stats.shapiro(residuals[:500])   # Shapiro-Wilk (max 5000 samples)
print(f'Shapiro-Wilk p-value: {p_shapiro:.4f}  '
      f'({"✗ non-normal" if p_shapiro < 0.05 else "✓ normal"})')

# ── 4. ACF / PACF (autocorrelation in residuals) ──────────────────────────────
# Compute on single pool to avoid cross-pool noise
pool_res = residuals[test['pool_id'].values == 'POOL-001']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(pool_res,  lags=24, ax=axes[0], title='ACF of Residuals')
plot_pacf(pool_res, lags=24, ax=axes[1], title='PACF of Residuals', method='ywm')
plt.tight_layout()

# Durbin-Watson (2 = no autocorrelation, <1.5 = positive autocorr)
dw = durbin_watson(pool_res)
print(f'Durbin-Watson: {dw:.3f}  (ideal ≈ 2.0, <1.5 = autocorrelation problem)')

# ── 5. Ljung-Box test (formal autocorrelation test) ───────────────────────────
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_result = acorr_ljungbox(pool_res, lags=[6, 12, 24], return_df=True)
print(lb_result)   # p > 0.05 → no significant autocorrelation at that lag

# ── 6. Heteroskedasticity (Breusch-Pagan) ────────────────────────────────────
from statsmodels.stats.diagnostic import het_breuschpagan
X_sm = sm.add_constant(y_pred)
_, bp_pvalue, _, _ = het_breuschpagan(residuals, X_sm)
print(f'Breusch-Pagan p-value: {bp_pvalue:.4f}  '
      f'({"✗ heteroskedastic" if bp_pvalue < 0.05 else "✓ homoskedastic"})')

# ── 7. Calibration (for classifiers) ─────────────────────────────────────────
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

prob_true, prob_pred = calibration_curve(y_test_clf, y_prob, n_bins=10)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(prob_pred, prob_true, 'o-', label='Model')
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration Curve')
ax.legend()

# Re-calibrate if needed (Platt scaling)
calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train_clf)
y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
```

---

## 11. Feature Importance

```python
from sklearn.inspection import permutation_importance
import pandas as pd

# ── 1. Tree-based: built-in importance (gain) ─────────────────────────────────
# XGBoost
fi_xgb = pd.DataFrame({
    'feature':    FEATURES,
    'importance': xgb_reg.feature_importances_   # 'weight' by default
}).sort_values('importance', ascending=False)

# Get different importance types from XGBoost
booster = xgb_reg.get_booster()
fi_weight = booster.get_score(importance_type='weight')    # num times feature used
fi_gain   = booster.get_score(importance_type='gain')      # avg gain per split
fi_cover  = booster.get_score(importance_type='cover')     # avg coverage per split
fi_df = pd.DataFrame({'weight': fi_weight, 'gain': fi_gain, 'cover': fi_cover}).fillna(0)
print(fi_df.sort_values('gain', ascending=False).head(10))

# LightGBM
fi_lgb = pd.DataFrame({
    'feature':    lgb_reg.feature_name_,
    'gain':       lgb_reg.feature_importances_
}).sort_values('gain', ascending=False)

# Random Forest
fi_rf = pd.DataFrame({
    'feature':    FEATURES,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

# ── 2. Permutation importance — model-agnostic, on TEST set ───────────────────
#       (preferred: shows actual predictive value, not training fit)
perm = permutation_importance(
    model,
    X_test, y_test_reg,
    n_repeats=10,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
fi_perm = pd.DataFrame({
    'feature':    FEATURES,
    'importance': perm.importances_mean,    # mean decrease in metric
    'std':        perm.importances_std
}).sort_values('importance', ascending=False)
print(fi_perm.head(10))

# Features with negative importance → may be adding noise
noise_features = fi_perm[fi_perm['importance'] < 0]['feature'].tolist()
print(f'Possibly noisy features: {noise_features}')

# ── 3. SHAP values (model-agnostic, shows direction of effect) ────────────────
import shap

# Tree-based models (fast exact computation)
explainer  = shap.TreeExplainer(xgb_reg)
shap_vals  = explainer.shap_values(X_test)           # shape: (n_samples, n_features)

shap.summary_plot(shap_vals, X_test, feature_names=FEATURES, plot_type='bar')  # importance
shap.summary_plot(shap_vals, X_test, feature_names=FEATURES)                   # beeswarm

# Single prediction explanation
shap.waterfall_plot(shap.Explanation(
    values=shap_vals[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=FEATURES
))

# ── 4. Plot importance comparison ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.barh(fi_xgb['feature'].head(12)[::-1],
        fi_xgb['importance'].head(12)[::-1], color='#D85A30')
ax.set_title('XGBoost Feature Importance (Gain)')

ax = axes[1]
ax.barh(fi_perm['feature'].head(12)[::-1],
        fi_perm['importance'].head(12)[::-1],
        xerr=fi_perm['std'].head(12)[::-1], color='#7F77DD')
ax.set_title('Permutation Importance (Test Set)')
plt.tight_layout()
```

---

## 12. Forecasting (ARIMA / Time Series)

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm       # pip install pmdarima

# ── Prepare single pool series ────────────────────────────────────────────────
pool_ts = df[df['pool_id'] == 'POOL-001'].sort_values('date').set_index('date')['cpr']
train_ts = pool_ts.iloc[:-12]   # leave last 12 months for testing
test_ts  = pool_ts.iloc[-12:]

# ── ACF/PACF to determine order ───────────────────────────────────────────────
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(train_ts,  lags=24, ax=axes[0])
plot_pacf(train_ts, lags=24, ax=axes[1], method='ywm')
# PACF cuts off at lag k → AR(k); ACF cuts off → MA(k)

# ── Manual ARIMA ──────────────────────────────────────────────────────────────
model_arima = ARIMA(train_ts, order=(2, 0, 1))   # (p, d, q)
fit_arima   = model_arima.fit()
print(fit_arima.summary())

forecast_result = fit_arima.get_forecast(steps=len(test_ts))
fc_mean = forecast_result.predicted_mean
fc_ci   = forecast_result.conf_int(alpha=0.05)

# ── Auto ARIMA (selects best p,d,q automatically) ─────────────────────────────
auto_model = pm.auto_arima(
    train_ts,
    start_p=0, max_p=4,
    start_q=0, max_q=4,
    d=None,                    # auto-detect differencing order
    seasonal=True, m=12,       # monthly seasonality
    information_criterion='aic',
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
print(auto_model.summary())
print(f'Best order: {auto_model.order}  seasonal: {auto_model.seasonal_order}')

# ── SARIMAX (with exogenous variables) ────────────────────────────────────────
exog_train = train[train['pool_id'] == 'POOL-001'][['refi_incentive', 'rate_10y']]
exog_test  = test[test['pool_id'] == 'POOL-001'][['refi_incentive', 'rate_10y']]

sarimax_model = SARIMAX(
    train_ts,
    exog=exog_train,
    order=(2, 0, 1),
    seasonal_order=(1, 0, 1, 12)
)
sarimax_fit = sarimax_model.fit(disp=False)
sarimax_fc  = sarimax_fit.get_forecast(steps=len(test_ts), exog=exog_test)

# ── Evaluate forecast ─────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse_arima = np.sqrt(mean_squared_error(test_ts, fc_mean))
mae_arima  = mean_absolute_error(test_ts, fc_mean)
naive_rmse = np.sqrt(mean_squared_error(test_ts.values[1:], test_ts.values[:-1]))

print(f'ARIMA RMSE: {rmse_arima:.3f}')
print(f'ARIMA MAE:  {mae_arima:.3f}')
print(f'Naïve RMSE: {naive_rmse:.3f}')

# ── Plot forecast ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train_ts.index, train_ts, 'b-', linewidth=1.5, label='Train')
ax.plot(test_ts.index,  test_ts,  'k-', linewidth=2,   label='Actual')
ax.plot(fc_mean.index,  fc_mean,  'r--',linewidth=2,   label='Forecast')
ax.fill_between(fc_ci.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                alpha=0.2, color='red', label='95% CI')
ax.axvline(test_ts.index[0], color='gray', linestyle=':', linewidth=1.5)
ax.set_ylabel('CPR (%)'); ax.legend()
ax.set_title('ARIMA Forecast vs Actual')
```

---

## 13. Stationarity & Decomposition

```python
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# ── ADF test ──────────────────────────────────────────────────────────────────
#    H0: unit root (non-stationary). Reject H0 → stationary.
def adf_report(series, name='Series'):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'\nADF Test: {name}')
    print(f'  Statistic : {result[0]:.4f}')
    print(f'  p-value   : {result[1]:.4f}')
    print(f'  Lags used : {result[2]}')
    for k, v in result[4].items():
        print(f'  Critical {k}: {v:.3f}')
    print(f'  → {"Stationary ✓" if result[1] < 0.05 else "Non-stationary ✗ (consider differencing)"}')
    return result[1]

adf_report(pool_ts, 'CPR (levels)')
adf_report(pool_ts.diff().dropna(), 'ΔCPR (first difference)')

# ── KPSS test ─────────────────────────────────────────────────────────────────
#    H0: stationary. Reject H0 → non-stationary.
#    Useful as complement to ADF (different null hypothesis)
kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(pool_ts.dropna(), regression='c', nlags='auto')
print(f'KPSS p-value: {kpss_p:.4f}  (reject H0 if < 0.05 → non-stationary)')

# ── Seasonal decomposition ────────────────────────────────────────────────────
decomp = seasonal_decompose(pool_ts, model='additive', period=12, extrapolate_trend='freq')

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for ax, comp, label in zip(axes,
    [pool_ts, decomp.trend, decomp.seasonal, decomp.resid],
    ['Observed','Trend','Seasonal','Residual']):
    ax.plot(comp, linewidth=1.5)
    ax.set_ylabel(label)
plt.suptitle('Seasonal Decomposition (additive)')
plt.tight_layout()

# ── STL decomposition (more robust — handles outliers better) ─────────────────
stl = STL(pool_ts, period=12, robust=True)
stl_result = stl.fit()
stl_result.plot()
plt.suptitle('STL Decomposition')

# ── Seasonal strength ─────────────────────────────────────────────────────────
seasonal_strength = 1 - np.var(stl_result.resid) / np.var(stl_result.seasonal + stl_result.resid)
trend_strength    = 1 - np.var(stl_result.resid) / np.var(stl_result.trend   + stl_result.resid)
print(f'Seasonal strength: {seasonal_strength:.3f}  (>0.6 = strong seasonality)')
print(f'Trend strength:    {trend_strength:.3f}')
```

---

## 14. Common Pitfalls

| Pitfall | Correct Approach |
|---|---|
| Random train/test split | Always split by time — temporal leakage destroys evaluation |
| Lags computed on full dataset before split | Compute lags per pool on training data only |
| Forgetting to group by pool | `.groupby('pool_id')['cpr'].shift(1)` — always |
| S-curve ignored | Linear on `refi_incentive` underfits — always add `**2` term |
| No seasonality | Summer CPR peak ~+2.5pp — always add `month_sin`/`month_cos` |
| Burnout omitted | Pools with low factor respond less to rate drops |
| Reporting R² alone | Add RMSE vs naïve baseline — a model can look good on R² but fail to beat naïve |
| Class imbalance ignored | Use `class_weight='balanced'` and report F1, not just accuracy |
| `TimeSeriesSplit` in `StackingRegressor` | Use `cv=5` (integer) — `TimeSeriesSplit` raises `cross_val_predict` error |
| OAS vs Z-spread | For pricing: always use OAS — Z-spread ignores prepay optionality |
| Single CPR assumption | CPR changes with rates — always state the prepayment speed assumption |
| Direction task AUC of 0.60 "is bad" | Near-random base rate — 0.60+ is already meaningful |
| ptp() deprecated | Use `np.ptp(arr)` or `arr.max() - arr.min()` on pandas Series |

---

## 15. Interview Presentation Framework

### Opening any analysis task

```
1. RESTATE   "The task is to predict next-month CPR for each MBS pool,
              using pool characteristics and market rate features."

2. QUALITY   "First I check for nulls, outliers, and date coverage per pool.
              Key concern: lag features introduce NaNs at pool start — drop after engineering."

3. SIGNAL    "The dominant predictor is refi_incentive = WAC − 10Y rate.
              The relationship is nonlinear (S-curve), so I add the squared term."

4. MODEL     "I start with Ridge as the interpretable baseline, then tree models
              for the nonlinear structure, and a Stacking ensemble to combine them."

5. METRICS   "For regression: RMSE vs naïve baseline — R² alone can mislead.
              For classification: AUC-ROC (ranking quality) and F1 (handles imbalance)."

6. RESULTS   "Best model: LightGBM, RMSE = X.XX vs naïve Y.YY — Z% improvement.
              Top features: refi_incentive, cpr_lag1, burnout."

7. LIMITS    "Assumes CPR is the right proxy for prepayment risk.
              No macro shocks (yield curve inversion, credit events) in features."

8. NEXT      "With more time: add yield curve features (2s10s spread, SOFR),
              MBS issuance volume, and housing market indicators (HPA, inventory)."
```

### Key numbers to know cold

```python
# These should be in your head before the interview
print('CPR range:        typical 5–40%, refi boom up to 60%')
print('WAC spread:       WAC ≈ market rate + 150–200bps (servicing + profit)')
print('LTV kink:         refi nearly impossible above 80% without extra insurance')
print('Seasonality:      +2.5pp CPR in summer (Jun–Aug) vs winter')
print('Agency guarantee: Fannie Mae, Freddie Mac pool the default risk')
print('OAS vs Z-spread:  OAS strips out option value; Z-spread does not')
print('TBA market:       ~$300B/day — most liquid fixed income market after Treasuries')
print('Burnout:          pools with factor < 0.3 respond ~half as much to rate drops')
```

---

*MBS Quant Research Cheatsheet · Squarepoint Capital Interview Preparation*
