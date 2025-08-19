# ==== PyTorch seq2seq with ROBUST preprocessing (median impute + safe standardize + safe MAE) ====
# Minimal change: wrap original script into a callable function that accepts only a delivery date.
# Everything else (logic, constants, SQL, architecture) remains the same.

import duckdb, pandas as pd, numpy as np, datetime as dt, math
import torch, torch.nn as nn
from sklearn.metrics import mean_absolute_error
from typing import Optional, Callable

# --- original constants kept ---
DB = "../ProjectMain/db/data.duckdb"
DELIVERY_DATE = pd.Timestamp("2025-08-18").date()   # default; overridden by function arg
W_PAST = 168
H_FUT = 24
HOLDOUT_DAYS = 60
HIDDEN = 128
EPOCHS = 30
LR = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(42)
torch.manual_seed(42)

# ---------- helpers  ----------
def fit_standardizer(X2d):
    """Return (mean,std) with std guarded (==0 -> 1). X2d: (N, F)"""
    mu = np.nanmean(X2d, axis=0)
    sd = np.nanstd(X2d, axis=0)
    sd = np.where(sd <= 1e-8, 1.0, sd)
    return mu, sd

def apply_standardizer(X, mu, sd):
    return (X - mu) / sd

# robust 3D imputer
def impute_inplace_3d(X, med_vec):
    """X: (N, T, F) or (N, W, F); med_vec: (F,). Fills NaNs in-place with per-feature medians."""
    mask = np.isnan(X)
    if mask.any():
        med_broadcast = np.broadcast_to(med_vec, X.shape)
        X[mask] = med_broadcast[mask]

# sanity checker
def assert_finite(name, arr):
    if not np.isfinite(arr).all():
        bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
        raise RuntimeError(f"{name} has {bad} non-finite values")

# Signed log transforms for spikes 
sgn = np.sign
abs_ = np.abs

def sgn_log1p(y):
    return sgn(y) * np.log1p(abs_(y))

def sgn_expm1(z):
    return sgn(z) * np.expm1(abs_(z))

# Hour-of-day weights (emphasize 17-21 local hours) — 
HOUR_WEIGHTS = np.ones(24, dtype=np.float32)
HOUR_WEIGHTS[:] = 1.0
HOUR_WEIGHTS[15] = 1.5     # H16
HOUR_WEIGHTS[16:18] = 2.0  # H17-18
HOUR_WEIGHTS[18:20] = 5.0  # H19-20 (peak focus)
HOUR_WEIGHTS[20] = 3.0     # H21

# ---------- model classes  ----------
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
    def forward(self, x):  # x: (B,W,E)
        _, h = self.rnn(x)
        return h  # (1,B,H)

class Decoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.rnn = nn.GRU(in_dim + 1, hidden, batch_first=True)  # + prev y
        self.out = nn.Linear(hidden, 1)
    def forward(self, future_feats, h0, y0, teacher=None, tf_prob=0.5):
        B, T, D = future_feats.shape
        y_prev = y0.view(B,1)
        h = h0
        outs = []
        for t in range(T):
            x_t = torch.cat([future_feats[:,t,:], y_prev], dim=1).unsqueeze(1)  # (B,1,D+1)
            o, h = self.rnn(x_t, h)
            y_t = self.out(o[:, -1, :]).squeeze(1)
            outs.append(y_t)
            if (teacher is not None) and (rng.random() < tf_prob):
                y_prev = teacher[:,t].view(B,1)
            else:
                y_prev = y_t.view(B,1)
        return torch.stack(outs, dim=1)  # (B,24)

class Seq2Seq(nn.Module):
    def __init__(self, enc_in, dec_in, hidden):
        super().__init__()
        self.enc = Encoder(enc_in, hidden)
        self.dec = Decoder(dec_in, hidden)
    def forward(self, x_enc, x_dec, y0, y_teacher=None, tf_prob=0.5):
        h = self.enc(x_enc)
        return self.dec(x_dec, h, y0, y_teacher, tf_prob)

# ---------- NEW THIN WRAPPER FUNCTION ----------
def run_deep_learning_forecast(
    delivery_date: "str | dt.date",
    db_path: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    epochs: Optional[int] = None,
):
    """
    Minimal wrapper around the original script so the UI can call it directly.
    - delivery_date: 'yyyy-mm-dd' or datetime.date
    - db_path: optional DuckDB path; defaults to original DB constant
    - log_fn: optional logger callback; if provided, epoch lines & messages go here
    - epochs: optional override for EPOCHS; otherwise uses the original constant

    Returns: pandas.DataFrame with columns [OperatingDTM, Interval, hb_houston_pred]
    """
    # ---- tiny helper to preserve/redirect prints ----
    def _log(msg: str):
        if log_fn is not None:
            try:
                log_fn(str(msg))
                return
            except Exception:
                pass
        print(msg)

    # ---- use original globals unless user overrides ----
    global DB, DELIVERY_DATE, EPOCHS
    DB_LOCAL = db_path if db_path else DB
    if isinstance(delivery_date, str):
        DELIVERY_DATE_LOCAL = pd.to_datetime(delivery_date).date()
    elif isinstance(delivery_date, dt.date):
        DELIVERY_DATE_LOCAL = delivery_date
    else:
        raise ValueError("delivery_date must be 'yyyy-mm-dd' or datetime.date")
    EPOCHS_LOCAL = int(epochs) if epochs is not None else EPOCHS

    # ---------- ensure weather-enhanced views (hist + fcst) exist  ----------
    con_ensure = duckdb.connect(DB_LOCAL)
    con_ensure.execute("""
    CREATE OR REPLACE VIEW wx_hist_enh AS
    WITH agg AS (
      SELECT
        OperatingDTM, "interval" AS Interval,
        (hist_temp_the_woodlands_tx + hist_temp_katy_tx + hist_temp_friendswood_tx + hist_temp_baytown_tx + hist_temp_houston_tx)/5.0 AS temp_avg,
        (hist_hum_the_woodlands_tx  + hist_hum_katy_tx  + hist_hum_friendswood_tx  + hist_hum_baytown_tx  + hist_hum_houston_tx)/5.0  AS hum_avg,
        GREATEST(hist_temp_the_woodlands_tx, hist_temp_katy_tx, hist_temp_friendswood_tx, hist_temp_baytown_tx, hist_temp_houston_tx)
          - LEAST(hist_temp_the_woodlands_tx, hist_temp_katy_tx, hist_temp_friendswood_tx, hist_temp_baytown_tx, hist_temp_houston_tx) AS temp_spread,
        GREATEST(hist_hum_the_woodlands_tx, hist_hum_katy_tx, hist_hum_friendswood_tx, hist_hum_baytown_tx, hist_hum_houston_tx)
          - LEAST(hist_hum_the_woodlands_tx, hist_hum_katy_tx, hist_hum_friendswood_tx, hist_hum_baytown_tx, hist_hum_houston_tx) AS hum_spread
      FROM vw_historical_weather_by_city
    )
    SELECT
      a.*,
      a.temp_avg - LAG(a.temp_avg) OVER (ORDER BY OperatingDTM, Interval) AS temp_avg_ramp1,
      a.hum_avg  - LAG(a.hum_avg)  OVER (ORDER BY OperatingDTM, Interval) AS hum_avg_ramp1
    FROM agg a;
    """)
    con_ensure.close()

    # ---------- pull data  ----------
    con = duckdb.connect(DB_LOCAL)
    df_lags = con.execute("""
    SELECT
      ts, OperatingDTM, Interval, hb_houston,
      p_lag1,p_lag2,p_lag3,p_lag6,p_lag12,p_lag24,p_lag48,p_lag72,p_lag168,
      dp1,dp24,p_roll24_mean,p_roll24_std,p_roll72_mean,p_roll168_mean
    FROM vw_master_spine_lags
    ORDER BY ts
    """).df()
    df_lags["ts"] = pd.to_datetime(df_lags["ts"])

    df_future = con.execute("""
    SELECT
      f.OperatingDTM AS delivery_date,
      f.Interval     AS delivery_interval,
      f.hb_houston   AS target,
      -- base future features
      f.wz_southcentral, f.wz_east, f.wz_west, f.wz_northcentral, f.wz_farwest, f.wz_north, f.wz_southern, f.wz_coast,
      f.lz_north, f.lz_west, f.lz_south, f.lz_houston,
      f.cal_hour AS cal_hour_f, f.cal_dow AS cal_dow_f, f.cal_is_weekend AS cal_is_weekend_f,
      f.cal_sin_hour AS cal_sin_hour_f, f.cal_cos_hour AS cal_cos_hour_f,
      f.cal_sin_dow  AS cal_sin_dow_f,  f.cal_cos_dow  AS cal_cos_dow_f,
      -- load-enhanced joins
      l.net_wz, l.net_lz, l.net_wz_ramp1, l.net_lz_ramp1, l.lz_houston_ramp1, l.wz_spread, l.lz_spread,
      -- weather: COALESCE forecast -> historical
      COALESCE(wf.temp_avg,      wh.temp_avg)      AS temp_avg,
      COALESCE(wf.temp_spread,   wh.temp_spread)   AS temp_spread,
      COALESCE(wf.temp_avg_ramp1,wh.temp_avg_ramp1)AS temp_avg_ramp1,
      COALESCE(wf.hum_avg,       wh.hum_avg)       AS hum_avg,
      COALESCE(wf.hum_spread,    wh.hum_spread)    AS hum_spread,
      COALESCE(wf.hum_avg_ramp1, wh.hum_avg_ramp1) AS hum_avg_ramp1,
      CASE WHEN wf.temp_avg IS NULL THEN 1 ELSE 0 END AS is_weather_proxy,
      -- additional ramps & day-over-day deltas to help with peaks (computed over time order)
      (l.net_wz      - LAG(l.net_wz,      3)  OVER (ORDER BY f.OperatingDTM, f.Interval)) AS net_wz_ramp3,
      (l.net_wz      - LAG(l.net_wz,      6)  OVER (ORDER BY f.OperatingDTM, f.Interval)) AS net_wz_ramp6,
      (l.net_lz      - LAG(l.net_lz,      3)  OVER (ORDER BY f.OperatingDTM, f.Interval)) AS net_lz_ramp3,
      (l.net_lz      - LAG(l.net_lz,      6)  OVER (ORDER BY f.OperatingDTM, f.Interval)) AS net_lz_ramp6,
      (l.lz_houston  - LAG(l.lz_houston,  3)  OVER (ORDER BY f.OperatingDTM, f.Interval)) AS lz_houston_ramp3,
      (l.lz_houston  - LAG(l.lz_houston,  6)  OVER (ORDER BY f.OperatingDTM, f.Interval)) AS lz_houston_ramp6,
      (l.net_wz      - LAG(l.net_wz,     24)  OVER (ORDER BY f.OperatingDTM, f.Interval)) AS net_wz_dod,
      (l.net_lz      - LAG(l.net_lz,     24)  OVER (ORDER BY f.OperatingDTM, f.Interval)) AS net_lz_dod,
      (l.lz_houston  - LAG(l.lz_houston, 24)  OVER (ORDER BY f.OperatingDTM, f.Interval)) AS lz_houston_dod
    FROM vw_master_spine_ts f
    LEFT JOIN load_fcst_enh l
      ON l.OperatingDTM = f.OperatingDTM AND l.Interval = f.Interval
    LEFT JOIN wx_fcst_enh  wf
      ON wf.OperatingDTM = f.OperatingDTM AND wf.Interval = f.Interval
    LEFT JOIN wx_hist_enh  wh
      ON wh.OperatingDTM = f.OperatingDTM AND wh.Interval = f.Interval
    ORDER BY f.OperatingDTM, f.Interval
    """).df()
    con.close()

    df_future["delivery_date"] = pd.to_datetime(df_future["delivery_date"]).dt.date
    df_future["delivery_interval"] = df_future["delivery_interval"].astype(int)

    # ---------- assemble samples (day-ahead)  ----------
    enc_cols = [
        "hb_houston","p_lag1","p_lag2","p_lag3","p_lag6","p_lag12","p_lag24","p_lag48","p_lag72","p_lag168",
        "dp1","dp24","p_roll24_mean","p_roll24_std","p_roll72_mean","p_roll168_mean"
    ]
    dec_future_cols = [
        "cal_hour_f","cal_dow_f","cal_is_weekend_f","cal_sin_hour_f","cal_cos_hour_f","cal_sin_dow_f","cal_cos_dow_f",
        "net_wz","net_lz","net_wz_ramp1","net_lz_ramp1","lz_houston_ramp1","wz_spread","lz_spread",
        "temp_avg","temp_spread","temp_avg_ramp1","hum_avg","hum_spread","hum_avg_ramp1",
        # NEW: multi-horizon ramps + day-over-day deltas
        "net_wz_ramp3","net_wz_ramp6","net_lz_ramp3","net_lz_ramp6",
        "lz_houston_ramp3","lz_houston_ramp6","net_wz_dod","net_lz_dod","lz_houston_dod",
        # Identify when weather came from historical backfill instead of forecast
        "is_weather_proxy"
    ]

    base_rows = df_lags[(df_lags["hb_houston"].notna()) & (df_lags["Interval"]==24)][["ts","OperatingDTM","hb_houston"]].copy()
    base_rows["base_date"] = pd.to_datetime(base_rows["OperatingDTM"]).dt.date
    base_rows["delivery_date"] = base_rows["base_date"] + dt.timedelta(days=1)

    has24 = df_future.groupby("delivery_date")["target"].apply(lambda s: s.notna().sum()==24)
    valid_delivery_dates = set(has24[has24].index)

    df_lags_idx = df_lags.set_index("ts").sort_index()

    samples = []
    for _, r in base_rows.iterrows():
        ddate = r["delivery_date"]
        ts0 = r["ts"]
        # encoder window
        start = ts0 - pd.Timedelta(hours=W_PAST-1)
        enc_df = df_lags_idx.loc[start:ts0][enc_cols]
        if enc_df.shape[0] != W_PAST:
            continue
        if enc_df.isna().any().any():
            continue  # strict: skip any encoder NaNs
        # decoder future rows
        fdf = df_future[(df_future["delivery_date"]==ddate)].sort_values("delivery_interval")
        if fdf.shape[0] != H_FUT:
            continue
        X_enc = enc_df.values.astype(np.float32)
        X_dec = fdf[dec_future_cols].values.astype(np.float32)
        y     = fdf["target"].values.astype(np.float32)
        y0    = np.float32(r["hb_houston"])
        samples.append((X_enc, X_dec, y, y0, ddate))

    if len(samples) == 0:
        raise RuntimeError("No samples assembled; check data coverage or reduce W_PAST.")

    dates = sorted({s[4] for s in samples})
    cut_date = dates[-1] - dt.timedelta(days=HOLDOUT_DAYS) if len(dates)>HOLDOUT_DAYS else dates[0]
    train_idx = [i for i,s in enumerate(samples) if s[4] <= cut_date and s[4] in valid_delivery_dates]
    val_idx   = [i for i,s in enumerate(samples) if s[4] >  cut_date and s[4] in valid_delivery_dates]

    def stack(idxs):
        Xe = np.stack([samples[i][0] for i in idxs], axis=0)
        Xd = np.stack([samples[i][1] for i in idxs], axis=0)
        Y  = np.stack([samples[i][2] for i in idxs], axis=0)
        Y0 = np.stack([samples[i][3] for i in idxs], axis=0)
        return Xe, Xd, Y, Y0

    Xe_tr, Xd_tr, Y_tr, Y0_tr = stack(train_idx)
    Xe_va, Xd_va, Y_va, Y0_va = (None, None, None, None)
    if len(val_idx) > 0:
        Xe_va, Xd_va, Y_va, Y0_va = stack(val_idx)

    # ---------- diagnose + drop all-NaN decoder features, then impute (robust) ----------
    F_orig = Xd_tr.shape[2]
    Xd_tr_2d_raw = Xd_tr.reshape(-1, F_orig)
    nan_counts = np.isnan(Xd_tr_2d_raw).sum(axis=0)
    all_nan_cols = nan_counts == Xd_tr_2d_raw.shape[0]

    if all_nan_cols.any():
        dropped_cols = [dec_future_cols[i] for i,flag in enumerate(all_nan_cols) if flag]
        _log(f"Dropping {int(all_nan_cols.sum())} decoder feature(s) with all-NaN in TRAIN: {dropped_cols}")
    else:
        dropped_cols = []

    dec_keep_mask = ~all_nan_cols

    Xd_tr = Xd_tr[:, :, dec_keep_mask]
    if Xe_va is not None:
        Xd_va = Xd_va[:, :, dec_keep_mask]

    dec_med = np.nanmedian(Xd_tr.reshape(-1, Xd_tr.shape[2]), axis=0)
    dec_med = np.where(np.isnan(dec_med), 0.0, dec_med)

    enc_med = np.nanmedian(Xe_tr.reshape(-1, Xe_tr.shape[2]), axis=0)
    enc_med = np.where(np.isnan(enc_med), 0.0, enc_med)

    impute_inplace_3d(Xe_tr, enc_med)
    impute_inplace_3d(Xd_tr, dec_med)
    if Xe_va is not None:
        impute_inplace_3d(Xe_va, enc_med)
        impute_inplace_3d(Xd_va, dec_med)

    assert np.isfinite(Y_tr).all(), "Y_tr has non-finite values"
    if Y_va is not None:
        assert np.isfinite(Y_va).all(), "Y_va has non-finite values"

    # ---------- safe standardization ----------
    enc_mu, enc_sd = fit_standardizer(Xe_tr.reshape(-1, Xe_tr.shape[2]))
    dec_mu, dec_sd = fit_standardizer(Xd_tr.reshape(-1, Xd_tr.shape[2]))
    Y_tr_t  = sgn_log1p(Y_tr)
    y_mu, y_sd = fit_standardizer(Y_tr_t.reshape(-1,1))
    y_mu = float(np.atleast_1d(y_mu)[0]); y_sd = float(np.atleast_1d(y_sd)[0])

    Xe_tr_n = apply_standardizer(Xe_tr, enc_mu, enc_sd)
    Xd_tr_n = apply_standardizer(Xd_tr, dec_mu, dec_sd)
    Y_tr_n  = apply_standardizer(Y_tr_t, y_mu, y_sd).reshape(Y_tr.shape)
    Y0_tr_t = sgn_log1p(Y0_tr.reshape(-1,1)).reshape(-1,1)
    Y0_tr_n = apply_standardizer(Y0_tr_t, y_mu, y_sd).reshape(-1)

    if Xe_va is not None:
        Xe_va_n = apply_standardizer(Xe_va, enc_mu, enc_sd)
        Xd_va_n = apply_standardizer(Xd_va, dec_mu, dec_sd)
        Y_va_t  = sgn_log1p(Y_va)
        Y_va_n  = apply_standardizer(Y_va_t,  y_mu,  y_sd).reshape(Y_va.shape)
        Y0_va_t = sgn_log1p(Y0_va.reshape(-1,1))
        Y0_va_n = apply_standardizer(Y0_va_t, y_mu, y_sd).reshape(-1)

    assert_finite("Xe_tr_n", Xe_tr_n)
    assert_finite("Xd_tr_n", Xd_tr_n)
    assert_finite("Y_tr_n",  Y_tr_n)
    assert_finite("Y0_tr_n", Y0_tr_n)
    if Xe_va is not None:
        assert_finite("Xe_va_n", Xe_va_n)
        assert_finite("Xd_va_n", Xd_va_n)
        assert_finite("Y_va_n",  Y_va_n)
        assert_finite("Y0_va_n", Y0_va_n)

    # ---------- torch datasets  ----------
    class Seq2SeqDataset(torch.utils.data.Dataset):
        def __init__(self, Xe, Xd, Y, Y0):
            self.Xe = torch.from_numpy(Xe).float()
            self.Xd = torch.from_numpy(Xd).float()
            self.Y  = torch.from_numpy(Y).float()
            self.Y0 = torch.from_numpy(Y0).float()
        def __len__(self): return self.Xe.shape[0]
        def __getitem__(self, i): return self.Xe[i], self.Xd[i], self.Y[i], self.Y0[i]

    bs = 32
    train_loader = torch.utils.data.DataLoader(Seq2SeqDataset(Xe_tr_n, Xd_tr_n, Y_tr_n, Y0_tr_n), batch_size=bs, shuffle=True)
    val_loader = None if Xe_va is None else torch.utils.data.DataLoader(Seq2SeqDataset(Xe_va_n, Xd_va_n, Y_va_n, Y0_va_n), batch_size=bs, shuffle=False)

    enc_in = Xe_tr_n.shape[2]
    dec_in = Xd_tr_n.shape[2]
    model = Seq2Seq(enc_in, dec_in, HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def huber_weighted(yhat, y, w, delta=1.0):
        d = torch.abs(yhat - y)
        q = torch.clamp(d, max=delta)
        l = 0.5 * q**2 + delta * (d - q)
        return (l * w).mean()

    # ---- train (other than logging sink) ----
    for ep in range(1, (EPOCHS_LOCAL)+1):
        model.train()
        tot = 0.0
        for xenc, xdec, y, y0 in train_loader:
            xenc, xdec, y, y0 = xenc.to(DEVICE), xdec.to(DEVICE), y.to(DEVICE), y0.to(DEVICE)
            opt.zero_grad()
            yhat = model(xenc, xdec, y0, y_teacher=y, tf_prob=0.5)
            w = torch.tensor(HOUR_WEIGHTS, device=y.device).unsqueeze(0).expand_as(y)
            loss = huber_weighted(yhat, y, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xenc.size(0)
        tr = tot / len(train_loader.dataset)

        va = None
        if val_loader is not None:
            model.eval()
            tot = 0.0
            with torch.no_grad():
                for xenc, xdec, y, y0 in val_loader:
                    xenc, xdec, y, y0 = xenc.to(DEVICE), xdec.to(DEVICE), y.to(DEVICE), y0.to(DEVICE)
                    yhat = model(xenc, xdec, y0, y_teacher=None, tf_prob=0.0)
                    w = torch.tensor(HOUR_WEIGHTS, device=y.device).unsqueeze(0).expand_as(y)
                    tot += huber_weighted(yhat, y, w).item() * xenc.size(0)
            va = tot / len(val_loader.dataset)

        _log(f"Epoch {ep:02d}  train_loss={tr:.4f}" + (f"  val_loss={va:.4f}" if va is not None else ""))

    # ---- evaluate holdout MAE in real units (SAFE) ----
    if val_loader is not None:
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xenc, xdec, y, y0 in val_loader:
                xenc, xdec, y, y0 = xenc.to(DEVICE), xdec.to(DEVICE), y.to(DEVICE), y0.to(DEVICE)
                yhat_n = model(xenc, xdec, y0, y_teacher=None, tf_prob=0.0).detach().cpu().numpy()
                y_n    = y.detach().cpu().numpy()
                yhat_t = (yhat_n * y_sd + y_mu)
                yt_t   = (y_n    * y_sd + y_mu)
                yhat = sgn_expm1(yhat_t).reshape(-1)
                yt   = sgn_expm1(yt_t).reshape(-1)
                preds.append(yhat)
                trues.append(yt)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mask = np.isfinite(preds) & np.isfinite(trues)
        dropped = int((~mask).sum())
        if dropped:
            _log(f"Dropped {dropped} non-finite pairs from holdout scoring.")
        _log({"holdout_MAE": float(mean_absolute_error(trues[mask], preds[mask]))})

    # ---- optional LightGBM residual booster  ----
    USE_LGBM_RESIDUAL = True
    booster = None
    if USE_LGBM_RESIDUAL:
        try:
            import lightgbm as lgb
            train_loader_eval = torch.utils.data.DataLoader(
                Seq2SeqDataset(Xe_tr_n, Xd_tr_n, Y_tr_n, Y0_tr_n), batch_size=64, shuffle=False)
            base_preds = []
            with torch.no_grad():
                model.eval()
                for xenc, xdec, y, y0 in train_loader_eval:
                    xenc, xdec, y, y0 = xenc.to(DEVICE), xdec.to(DEVICE), y.to(DEVICE), y0.to(DEVICE)
                    yhat_n = model(xenc, xdec, y0, y_teacher=None, tf_prob=0.0).detach().cpu().numpy()
                    base_preds.append(yhat_n)
            base_preds = np.concatenate(base_preds, axis=0)
            base_preds_t = (base_preds * y_sd + y_mu)
            base_preds_true = sgn_expm1(base_preds_t)
            y_true_tr = Y_tr
            y_base_flat = base_preds_true.reshape(-1)
            y_true_flat = y_true_tr.reshape(-1)
            X_flat = Xd_tr.reshape(-1, Xd_tr.shape[2])
            mask = np.isfinite(y_base_flat) & np.isfinite(y_true_flat) & np.isfinite(X_flat).all(axis=1)
            X_flat = X_flat[mask]
            y_resid_flat = (y_true_flat - y_base_flat)[mask]
            booster = lgb.LGBMRegressor(
                n_estimators=600, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                max_depth=-1, reg_alpha=0.0, reg_lambda=0.0, min_child_samples=20, random_state=42)
            booster.fit(X_flat, y_resid_flat)
            _log("Residual booster trained (LightGBM).")
        except Exception as e:
            _log(f"LightGBM residual booster skipped: {e}")

    # ---- predict requested DELIVERY_DATE  ----
    prev_day = DELIVERY_DATE_LOCAL - dt.timedelta(days=1)
    base_row = df_lags[(pd.to_datetime(df_lags["OperatingDTM"]).dt.date == prev_day) & (df_lags["Interval"]==24) & df_lags["hb_houston"].notna()]
    if base_row.empty:
        raise RuntimeError(f"No base EOD price for {prev_day}")
    ts0 = pd.to_datetime(base_row.iloc[0]["ts"])
    y0  = np.float32(base_row.iloc[0]["hb_houston"]) 

    win = df_lags.set_index("ts").loc[ts0 - pd.Timedelta(hours=W_PAST-1): ts0][enc_cols]
    if win.shape[0] != W_PAST or win.isna().any().any():
        raise RuntimeError("Insufficient/NaN history for encoder window.")
    Xe_pred = win.values.astype(np.float32)[None, ...]

    frows = df_future[(df_future["delivery_date"]==DELIVERY_DATE_LOCAL)].sort_values("delivery_interval")
    if frows.shape[0] != H_FUT:
        raise RuntimeError("Spine missing 24 rows for delivery date.")
    Xd_pred_full = frows[dec_future_cols].values.astype(np.float32)[None, ...]

    if 'dec_keep_mask' in locals():
        Xd_pred = Xd_pred_full[:, :, dec_keep_mask]
    else:
        Xd_pred = Xd_pred_full

    m = np.isnan(Xd_pred)
    if m.any():
        Xd_pred[m] = np.broadcast_to(dec_med, Xd_pred.shape)[m]
    Xe_pred_n = apply_standardizer(Xe_pred, enc_mu, enc_sd)
    Xd_pred_n = apply_standardizer(Xd_pred, dec_mu, dec_sd)
    y0_t = sgn_log1p(y0)
    y0_n = np.float32(((y0_t - y_mu) / y_sd)).ravel()[0]

    model.eval()
    with torch.no_grad():
        yhat_n = model(torch.from_numpy(Xe_pred_n).to(DEVICE),
                       torch.from_numpy(Xd_pred_n).to(DEVICE),
                       torch.tensor([y0_n], dtype=torch.float32, device=DEVICE),
                       y_teacher=None, tf_prob=0.0).detach().cpu().numpy()[0]
    yhat_t = (yhat_n * y_sd + y_mu).reshape(-1)
    yhat = sgn_expm1(yhat_t)

    # optional residual booster on peak hours
    if 'booster' in locals() and booster is not None:
        Xd_pred_kept = Xd_pred.copy()
        resid_pred = booster.predict(Xd_pred_kept.reshape(24, Xd_pred_kept.shape[2]))
        hour_mask = np.zeros(24, dtype=np.float32)
        hour_mask[17:21] = 1.0
        yhat = yhat + resid_pred * hour_mask

    if np.isnan(yhat).any():
        day_med = float(np.nanmedian(yhat))
        yhat = np.nan_to_num(yhat, nan=day_med)

    forecast = pd.DataFrame({
        "OperatingDTM": [DELIVERY_DATE_LOCAL]*H_FUT,
        "Interval": list(range(1, H_FUT+1)),
        "hb_houston_pred": yhat.tolist()
    })
    return forecast

# allow quick local testing when run directly
if __name__ == "__main__":
    out = run_deep_learning_forecast(DELIVERY_DATE, DB)
    print(out.head())


