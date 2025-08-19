from __future__ import annotations
from typing import Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

def _safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def fit_with_backtest(features_df: pd.DataFrame, target_col: str = "lmp") -> Dict[str, Any]:
    """Tiny rolling-style split + fit; replace with your full routine later."""
    df = features_df.copy()
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' missing")

    split = int(len(df) * 0.8)
    train, valid = df.iloc[:split], df.iloc[split:]

    X_cols = [c for c in df.columns if c not in {target_col, 'timestamp'}]
    X_train, y_train = train[X_cols], train[target_col]
    X_valid, y_valid = valid[X_cols], valid[target_col]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
        tree_method='hist',
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)

    rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))
    mae = float(mean_absolute_error(y_valid, preds))
    mape = _safe_mape(y_valid, preds)

    try:
        imp = model.get_booster().get_score(importance_type='gain')
        feat_imp = [{'feature': k, 'gain': float(v)} for k, v in sorted(imp.items(), key=lambda x: -x[1])]
    except Exception:
        feat_imp = [{'feature': c, 'gain': 0.0} for c in X_cols]

    return {
        'trained_model': 'xgboost_regressor',
        'metrics': {'rmse': round(rmse, 3), 'mae': round(mae, 3), 'mape': round(mape, 3)},
        'feature_importance': feat_imp,
        'generated_at': datetime.utcnow().isoformat(timespec='seconds'),
    }
