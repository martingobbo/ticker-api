# jobs/daily_runner.py
# =========================
# DAILY RUNNER (Azure Postgres + FMP)
# - Runs for a TARGET_DATE (arg or env), pulls OHLCV from FMP, computes metrics,
#   upserts prices + metrics, refreshes snapshot_metric_latest.
#
# SUPPORTED TICKER TYPES:
#   Stock, ETF, Index  — full metrics computed
#   FXPair             — prices only (close=adj_close, no volume, no metrics)
#   Futures            — prices only (close=adj_close, volume kept, no metrics)
#
# SECURITY:
# - NO secrets in code. Read from environment variables:
#   FMP_API_KEY, PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, (optional) PGSSLMODE
#
# IMPORTANT FIX:
# - DOES NOT create tables. snapshot_metric_latest must already exist.
#   (Remove CREATE TABLE IF NOT EXISTS to keep least-privilege runner user.)
# =========================

import os
import math
import time
import argparse
from datetime import date as _date
from typing import Optional, Dict, Any, List

import requests
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extras import execute_values


# -------------------------
# CLI / DATE
# -------------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-date", default=os.getenv("TARGET_DATE", ""), help="YYYY-MM-DD")
    return p.parse_args()


args = _parse_args()
if args.target_date:
    TARGET_DATE = args.target_date
else:
    TARGET_DATE = _date.today().isoformat()  # fallback


# -------------------------
# REQUIRED CONFIG
# -------------------------
SCHEMA = os.getenv("SCHEMA", "analytics")
ROWS_PER_TICKER = int(os.getenv("ROWS_PER_TICKER", "750"))

# -------------------------
# TYPE CLASSIFICATION (matches backfill script)
# -------------------------
NO_VOLUME_TYPES = {"FXPair", "Index"}
NO_ADJ_CLOSE_TYPES = {"FXPair", "Futures", "Index", "Crypto"}
METRICS_TICKER_TYPES = {"Stock", "ETF", "Index"}       # compute + store metrics
PRICE_ONLY_TYPES = {"FXPair", "Futures"}                # prices only, no metrics
FMP_TICKER_TYPES = {"Stock", "ETF", "Index", "FXPair", "Futures"}

# -------------------------
# API + DB CONFIG (ENV VARS ONLY)
# -------------------------
FMP_API_KEY = os.environ["FMP_API_KEY"]

PGHOST = os.environ["PGHOST"]
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "postgres")
PGUSER = os.environ["PGUSER"]
PGPASSWORD = os.environ["PGPASSWORD"]
PGSSLMODE = os.getenv("PGSSLMODE", "require")

# -------------------------
# CONNECT
# -------------------------
ENGINE_URL = (
    f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    f"?sslmode={PGSSLMODE}"
)
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

pg_conn = psycopg2.connect(
    host=PGHOST,
    port=PGPORT,
    dbname=PGDATABASE,
    user=PGUSER,
    password=PGPASSWORD,
    sslmode=PGSSLMODE,
)

T_DATE = pd.Timestamp(TARGET_DATE).date()


# -------------------------
# 1) LOAD TICKER UNIVERSE FROM AZURE
# -------------------------
dim_ticker = pd.read_sql(
    text(f"SELECT ticker_id, ticker, ticker_type FROM {SCHEMA}.dim_ticker WHERE ticker IS NOT NULL"),
    engine,
)

# Filter to FMP-relevant ticker types only
dim_ticker = dim_ticker[dim_ticker["ticker_type"].isin(FMP_TICKER_TYPES)].copy()

tickers = dim_ticker["ticker"].dropna().astype(str).str.strip()
tickers = tickers[tickers != ""].unique().tolist()

ticker_to_id = dict(zip(dim_ticker["ticker"], dim_ticker["ticker_id"]))
ticker_to_type = dict(zip(dim_ticker["ticker"], dim_ticker["ticker_type"]))

print(f"Tickers loaded from Azure (FMP types): {len(tickers):,}")


# -------------------------
# 2) DOWNLOAD DAILY OHLCV FROM FMP FOR TARGET_DATE (per-symbol, concurrent)
# -------------------------
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
SLEEP_BETWEEN_BATCHES = float(os.getenv("SLEEP_BETWEEN_BATCHES", "2.0"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1.5"))


def fetch_daily(symbol: str, date_from: str, date_to: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch daily bars for a symbol over [date_from, date_to] from FMP historical-price-full/{symbol}.
    Retries 429/5xx. Returns list[dict] (historical) or None on hard failure.
    """
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {"from": date_from, "to": date_to, "apikey": FMP_API_KEY}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE**attempt)
                continue
            return None

        if r.status_code in (429, 502, 503, 504):
            if attempt < MAX_RETRIES:
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else (BACKOFF_BASE**attempt)
                time.sleep(delay)
                continue
            return None

        if r.status_code != 200:
            return None

        try:
            js = r.json()
        except ValueError:
            return None

        return js.get("historical", [])

    return None


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


date_from = TARGET_DATE
date_to = TARGET_DATE

all_rows = []
no_data = []
hard_fail = []

batches = list(chunked(tickers, BATCH_SIZE))
for bi, batch in enumerate(batches, start=1):
    print(f"Batch {bi}/{len(batches)}: {len(batch)} symbols")

    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for sym in batch:
            futures[ex.submit(fetch_daily, sym, date_from, date_to)] = sym

        for fut in as_completed(futures):
            sym = futures[fut]
            hist = fut.result()
            if hist is None:
                hard_fail.append(sym)
                continue

            hist = [h for h in hist if h.get("date") == TARGET_DATE]

            if not hist:
                no_data.append(sym)
                continue

            ttype = ticker_to_type.get(sym, "Stock")

            for h in hist:
                close_val = h.get("close")
                # FXPair/Futures/Index/Crypto: no adjClose from FMP, use close
                adj_close_val = close_val if ttype in NO_ADJ_CLOSE_TYPES else h.get("adjClose")
                # FXPair/Index: no meaningful volume
                volume_val = None if ttype in NO_VOLUME_TYPES else h.get("volume")

                all_rows.append(
                    {
                        "date": h.get("date"),
                        "ticker": sym,
                        "open": h.get("open"),
                        "high": h.get("high"),
                        "low": h.get("low"),
                        "close": close_val,
                        "adjClose": adj_close_val,
                        "volume": volume_val,
                    }
                )

    if bi < len(batches):
        time.sleep(SLEEP_BETWEEN_BATCHES)

data = pd.DataFrame(all_rows)
if not data.empty:
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

fetched = data["ticker"].nunique() if not data.empty else 0
print(f"\nDone. Tickers requested: {len(tickers)}")
print(f"Tickers with rows returned: {fetched}")
print(f"Rows fetched: {len(data)}")

if no_data:
    preview = ", ".join(no_data[:20])
    more = f" ... (+{len(no_data)-20} more)" if len(no_data) > 20 else ""
    print("\nNo rows returned for the requested date/window:")
    print(preview + more)

if hard_fail:
    preview = ", ".join(hard_fail[:20])
    more = f" ... (+{len(hard_fail)-20} more)" if len(hard_fail) > 20 else ""
    print("\nFailed after retries (HTTP/network errors):")
    print(preview + more)

if data.empty:
    print(f"\n[STOP] FMP returned 0 rows for {TARGET_DATE}.")
    print("This usually means the date is a weekend/holiday, or FMP has no data for that day yet.")
    raise SystemExit(0)


# -------------------------
# 2b) SEPARATE price-only tickers (FXPair/Futures) from metrics tickers
# -------------------------
data_renamed = data.rename(columns={"date": "dt", "adjClose": "adj_close"}).copy()
data_renamed["dt"] = pd.to_datetime(data_renamed["dt"]).dt.date
data_renamed["ticker_type"] = data_renamed["ticker"].map(ticker_to_type)

data_renamed = data_renamed[["ticker", "ticker_type", "dt", "open", "high", "low", "close", "adj_close", "volume"]]

# Split into metrics vs price-only
data_metrics = data_renamed[data_renamed["ticker_type"].isin(METRICS_TICKER_TYPES)].copy()
data_price_only = data_renamed[data_renamed["ticker_type"].isin(PRICE_ONLY_TYPES)].copy()


# -------------------------
# 3) LOAD RECENT HISTORY FROM AZURE (fact_price_daily)
#    Only for metrics tickers — price-only types don't need history for metrics
# -------------------------
sql_prices = f"""
WITH ranked AS (
  SELECT
    f.ticker_id,
    t.ticker,
    t.ticker_type,
    f.dt,
    f.open, f.high, f.low, f.close, f.adj_close, f.volume,
    ROW_NUMBER() OVER (PARTITION BY f.ticker_id ORDER BY f.dt DESC) AS rn
  FROM {SCHEMA}.fact_price_daily AS f
  JOIN {SCHEMA}.dim_ticker       AS t ON t.ticker_id = f.ticker_id
  WHERE t.ticker_type IN ('Stock', 'ETF', 'Index')
)
SELECT
  ticker, ticker_type, dt, open, high, low, close, adj_close, volume
FROM ranked
WHERE rn <= :rows_per_ticker
ORDER BY ticker, dt
"""
prices = pd.read_sql(text(sql_prices), engine, params={"rows_per_ticker": ROWS_PER_TICKER})


# -------------------------
# 4) COMBINE TODAY WITH HISTORY (metrics tickers only)
# -------------------------
combined = pd.concat([prices, data_metrics], ignore_index=True)
combined = combined.sort_values(["ticker", "dt"]).reset_index(drop=True)


# -------------------------
# 5) METRICS CALCULATION
# -------------------------
TRADING_DAYS_PER_YEAR = 252
WIN_05, WIN_10, WIN_15, WIN_20, WIN_50 = 5, 10, 15, 20, 50
WIN_60, WIN_100, WIN_200, WIN_252, WIN_300, WIN_750 = 60, 100, 200, 252, 300, 750
SMA_POS_LEN = 3

# Maximum plausible daily volume — anything above this is treated as corrupt FMP data
MAX_VALID_VOLUME = 100_000_000_000  # 100 billion shares


def ols_slope_window(arr: np.ndarray) -> float:
    mask = np.isfinite(arr)
    y = arr[mask]
    n = y.size
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return np.nan
    num = np.sum((x - xm) * (y - ym))
    return float(num / denom)


def max_drawdown_only(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    if not np.isfinite(a).any():
        return np.nan
    start_idx = -1
    peak = np.nan
    for i in range(a.size):
        if math.isfinite(a[i]) and a[i] > 0.0:
            peak = a[i]
            start_idx = i
            break
    if start_idx < 0:
        return np.nan
    best_dd = 0.0
    for j in range(start_idx + 1, a.size):
        pj = a[j]
        if not math.isfinite(pj) or pj <= 0.0:
            continue
        if pj > peak:
            peak = pj
        dd = pj / peak - 1.0
        if dd < best_dd:
            best_dd = dd
    return float(best_dd)


def max_drawdown_duration_only(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    if not np.isfinite(a).any():
        return np.nan
    max_price = np.nan
    max_idx = -1
    for i in range(a.size):
        if math.isfinite(a[i]) and a[i] > 0.0:
            max_price = a[i]
            max_idx = i
            break
    if max_idx < 0:
        return np.nan
    best_dd = 0.0
    best_dur = 0
    for j in range(max_idx + 1, a.size):
        pj = a[j]
        if not math.isfinite(pj) or pj <= 0.0:
            continue
        if pj > max_price:
            max_price = pj
            max_idx = j
        dd = pj / max_price - 1.0
        if dd < best_dd:
            best_dd = dd
            best_dur = j - max_idx
    return float(best_dur)


def per_ticker_metrics_all_rows(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["date"] = pd.to_datetime(g["dt"])
    g = g.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    ttype = (str(g["ticker_type"].iloc[0]) if "ticker_type" in g.columns else "Stock").strip().lower()
    # Volume calculations apply to Stock and ETF; Index has no meaningful volume data
    uses_volume = ttype in ("stock", "etf")

    close = pd.to_numeric(g["adj_close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(g["open"], errors="coerce").astype(float)
    high = pd.to_numeric(g["high"], errors="coerce").astype(float)
    low = pd.to_numeric(g["low"], errors="coerce").astype(float)

    vol_original = (
        pd.to_numeric(g["volume"], errors="coerce").astype(float)
        if "volume" in g.columns
        else pd.Series(np.nan, index=g.index)
    )
    vol_calc = vol_original if uses_volume else pd.Series(np.nan, index=g.index)

    close_safe = close.replace(0, np.nan)
    log_price = np.log(close_safe)
    prev_close = close_safe.shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.where(
            (close_safe > 0) & (prev_close > 0), np.log(close_safe / prev_close), np.nan
        )
    log_returns = pd.Series(log_returns, index=g.index)
    if len(g) > 0 and pd.isna(log_returns.iloc[0]):
        log_returns.iloc[0] = 0.0

    ma20 = close.rolling(WIN_20, min_periods=1).mean()
    ma50 = close.rolling(WIN_50, min_periods=1).mean()
    ma100 = close.rolling(WIN_100, min_periods=1).mean()
    ma200 = close.rolling(WIN_200, min_periods=1).mean()

    dv = close * vol_calc
    sum_px_vol_20 = dv.rolling(WIN_20, min_periods=1).sum()
    sum_vol_20 = vol_calc.rolling(WIN_20, min_periods=1).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap20 = np.where(sum_vol_20 > 0, sum_px_vol_20 / sum_vol_20, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_dv = np.where((vol_calc > 0) & (close > 0), np.log(vol_calc * close), np.nan)
    ln_dv = pd.Series(ln_dv, index=g.index)
    vol_accel_5d = ln_dv - ln_dv.shift(5)
    vol_accel_10d = ln_dv - ln_dv.shift(10)

    avg10_dv = dv.rolling(WIN_10, min_periods=1).mean()
    avg60_dv = dv.rolling(WIN_60, min_periods=1).mean()
    std60_dv = dv.rolling(WIN_60, min_periods=2).std(ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        abn_vol_60d = np.where(std60_dv > 0, (avg10_dv - avg60_dv) / std60_dv, np.nan)

    vol20_ann = log_returns.rolling(WIN_20, min_periods=2).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol100_ann = log_returns.rolling(WIN_100, min_periods=2).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    mean100 = log_returns.rolling(WIN_100, min_periods=1).mean()

    low_10 = low.rolling(WIN_10, min_periods=1).min()
    high_10 = high.rolling(WIN_10, min_periods=1).max()
    rng_10 = high_10 - low_10
    with np.errstate(invalid="ignore", divide="ignore"):
        pos_10d = np.where(rng_10 != 0, (close - low_10) / rng_10, 0.0)
    pos_10d = pd.Series(pos_10d, index=g.index)
    five_day_range_pos = pos_10d.rolling(SMA_POS_LEN, min_periods=1).mean()

    daily_range = (high - low)
    avg_rng_10 = daily_range.rolling(WIN_10, min_periods=1).mean()
    avg_rng_60 = daily_range.rolling(WIN_60, min_periods=1).mean()
    std_rng_60 = daily_range.rolling(WIN_60, min_periods=2).std(ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_60_10_highlowrange = np.where(std_rng_60 > 0, (avg_rng_10 - avg_rng_60) / std_rng_60, 0.0)

    def safe_lr(curr, lagged):
        return np.where((curr > 0) & (lagged > 0), np.log(curr / lagged), 0.0)

    ret5 = safe_lr(close, close.shift(5))
    ret10 = safe_lr(close, close.shift(10))
    ret20 = safe_lr(close, close.shift(20))
    ret40 = safe_lr(close, close.shift(40))
    ret60 = safe_lr(close, close.shift(60))
    ret200 = safe_lr(close, close.shift(200))
    ret300 = safe_lr(close, close.shift(300))

    median_return_100d = log_returns.rolling(WIN_100, min_periods=1).median()

    closes_by_date = pd.Series(close.values, index=g["date"])

    def _dd_percent(w: pd.Series) -> float:
        v = pd.to_numeric(w, errors="coerce").to_numpy(dtype="float64")
        if v.size <= 1 or not np.isfinite(v).any():
            return 0.0
        try:
            peak_pos = np.nanargmax(v)
        except ValueError:
            return 0.0
        peak = v[peak_pos]
        if not np.isfinite(peak):
            return 0.0
        suffix = v[peak_pos:]
        if suffix.size == 0 or not np.isfinite(suffix).any():
            return 0.0
        trough = np.nanmin(suffix)
        return float(trough / peak - 1.0)

    def _dd_duration(w: pd.Series) -> float:
        v = pd.to_numeric(w, errors="coerce").to_numpy(dtype="float64")
        idx = w.index
        if v.size <= 1 or not np.isfinite(v).any():
            return 0.0
        try:
            peak_pos = np.nanargmax(v)
        except ValueError:
            return 0.0
        suffix = v[peak_pos:]
        if suffix.size == 0 or not np.isfinite(suffix).any():
            return 0.0
        trough_rel = np.nanargmin(suffix)
        trough_pos = peak_pos + trough_rel
        return float((idx[trough_pos] - idx[peak_pos]).days)

    drawdown_percent_100 = closes_by_date.rolling(100, min_periods=1).apply(_dd_percent, raw=False).values
    drawdown_days_100 = closes_by_date.rolling(100, min_periods=1).apply(_dd_duration, raw=False).values

    lr = pd.Series(log_returns, index=g.index)
    vol_5 = lr.rolling(WIN_05, min_periods=WIN_05).std(ddof=1)
    vol_15 = lr.rolling(WIN_15, min_periods=WIN_15).std(ddof=1)
    vol_60 = lr.rolling(WIN_60, min_periods=WIN_60).std(ddof=1)
    vol_252 = lr.rolling(WIN_252, min_periods=WIN_252).std(ddof=1)

    neg = np.minimum(lr, 0.0)
    pos = np.maximum(lr, 0.0)
    dd_15 = (pd.Series(neg).pow(2).rolling(WIN_15, min_periods=WIN_15).mean()) ** 0.5
    dd_60 = (pd.Series(neg).pow(2).rolling(WIN_60, min_periods=WIN_60).mean()) ** 0.5
    dd_252 = (pd.Series(neg).pow(2).rolling(WIN_252, min_periods=WIN_252).mean()) ** 0.5
    ud_15 = (pd.Series(pos).pow(2).rolling(WIN_15, min_periods=WIN_15).mean()) ** 0.5
    ud_60 = (pd.Series(pos).pow(2).rolling(WIN_60, min_periods=WIN_60).mean()) ** 0.5
    ud_252 = (pd.Series(pos).pow(2).rolling(WIN_252, min_periods=WIN_252).mean()) ** 0.5

    hl_log = np.log((high.replace(0, np.nan)) / (low.replace(0, np.nan)))
    k = 1.0 / (4.0 * math.log(2.0))
    pk20 = np.sqrt(k * (hl_log.pow(2).rolling(WIN_20, min_periods=WIN_20).mean()))

    sum10 = lr.rolling(WIN_10, min_periods=WIN_10).sum()
    change_10dayret = sum10 - sum10.shift(WIN_10)

    slope_lp_recent60 = (
        pd.Series(np.log(close.replace(0, np.nan)))
        .rolling(WIN_60, min_periods=WIN_60)
        .apply(ols_slope_window, raw=True)
    )
    slope_lp_prev60 = slope_lp_recent60.shift(WIN_60)
    ret_accel_60 = slope_lp_recent60 - slope_lp_prev60

    slope_vol60_over20 = vol_60.rolling(WIN_20, min_periods=WIN_20).apply(ols_slope_window, raw=True)
    slope_vol252_over60 = vol_252.rolling(WIN_60, min_periods=WIN_60).apply(ols_slope_window, raw=True)

    dv_sma_252 = dv.rolling(WIN_252, min_periods=WIN_252).mean()
    dv_sma_60 = dv.rolling(WIN_60, min_periods=WIN_60).mean()
    dv252_accel_60 = dv_sma_252.rolling(WIN_60, min_periods=WIN_60).apply(ols_slope_window, raw=True)
    corr_px_dv_60 = close.rolling(WIN_60, min_periods=WIN_60).corr(dv)

    ema5_of_vol15 = vol_15.ewm(span=5, adjust=False).mean()

    mdd_750 = close.rolling(WIN_750, min_periods=2).apply(max_drawdown_only, raw=True)
    mdd_dur_750 = close.rolling(WIN_750, min_periods=2).apply(max_drawdown_duration_only, raw=True)

    out = pd.DataFrame(
        {
            "date": g["date"].values,
            "ticker": g["ticker"].values,
            "ticker_type": g["ticker_type"].values,
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": pd.to_numeric(g["close"], errors="coerce").astype(float).values,
            "adj_close": close.values,
            "volume": vol_original.values,
            "log_returns": log_returns.values,
            "volatility_20d": vol20_ann.values,
            "volatility_100d": vol100_ann.values,
            "mean_return_100d": mean100.values,
            "moving_avg_20d": ma20.values,
            "moving_avg_50d": ma50.values,
            "moving_avg_100d": ma100.values,
            "moving_avg_200d": ma200.values,
            "vwap_20d": vwap20,
            "vol_accel_5d": vol_accel_5d.values,
            "vol_accel_10d": vol_accel_10d.values,
            "abn_vol_60d": abn_vol_60d,
            "5_day_range_pos": five_day_range_pos.values,
            "60_10_highlowrange_zscore": z_60_10_highlowrange,
            "5_day_ret": ret5,
            "10_day_ret": ret10,
            "20_day_ret": ret20,
            "40_day_ret": ret40,
            "60_day_ret": ret60,
            "200_day_ret": ret200,
            "300_day_ret": ret300,
            "median_return_100d": median_return_100d.values,
            "drawdown_percent": drawdown_percent_100,
            "drawdown_duration_days": drawdown_days_100,
            "log_prices": log_price.values,
            "change_10dayret": change_10dayret.values,
            "slope_over60_of_logprice": slope_lp_recent60.values,
            "prior_slope_over60_of_logprice": slope_lp_prev60.values,
            "60d_return_accel": ret_accel_60.values,
            "750d_drawdown": mdd_750.values,
            "750d_drawdownduration": mdd_dur_750.values,
            "15d_downsidedeviation": dd_15.values,
            "60d_downsidedeviation": dd_60.values,
            "252d_downsidedeviation": dd_252.values,
            "15d_upsidevolatility": ud_15.values,
            "60d_upsidevolatility": ud_60.values,
            "252d_upsidevolatility": ud_252.values,
            "5d_volatility": vol_5.values,
            "15d_volatility": vol_15.values,
            "60d_volatility": vol_60.values,
            "252d_volatility": vol_252.values,
            "20d_parkinson_HL_volatility": pk20.values,
            "5d_EMA_15dayvolatility": ema5_of_vol15.values,
            "slope_over20_of_60d_volatility": slope_vol60_over20.values,
            "slope_over60_of_252d_volatility": slope_vol252_over60.values,
            "252d_dollar_volume_SMA": dv_sma_252.values,
            "60d_dollar_volume_SMA": dv_sma_60.values,
            "252d_dollar_volume_accel": dv252_accel_60.values,
            "60d_price_dollarVolume_correlation": corr_px_dv_60.values,
        }
    )

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out[["5_day_range_pos", "60_10_highlowrange_zscore", "drawdown_percent", "drawdown_duration_days"]] = out[
        ["5_day_range_pos", "60_10_highlowrange_zscore", "drawdown_percent", "drawdown_duration_days"]
    ].fillna(0.0)

    return out


combined["dt"] = pd.to_datetime(combined["dt"]).dt.date

daily_chunks = []
for tkr, g in combined.groupby("ticker", sort=False):
    if not (pd.to_datetime(g["dt"]).dt.date == T_DATE).any():
        continue
    all_rows = per_ticker_metrics_all_rows(g)
    day_row = all_rows[pd.to_datetime(all_rows["date"]).dt.date == T_DATE]
    if not day_row.empty:
        daily_chunks.append(day_row)

daily_metrics = pd.concat(daily_chunks, ignore_index=True) if daily_chunks else pd.DataFrame()
print(f"Metrics rows computed for {T_DATE}: {len(daily_metrics):,}")


# -------------------------
# 6) MAP to ticker_id + metric_id
# -------------------------
dim_metric = pd.read_sql(text(f"SELECT metric_id, metric_code FROM {SCHEMA}.dim_metric"), engine)
metric_map = dict(zip(dim_metric["metric_code"], dim_metric["metric_id"]))

if not daily_metrics.empty:
    daily_metrics["ticker_id"] = daily_metrics["ticker"].map(ticker_to_id)
    unmapped = daily_metrics.loc[daily_metrics["ticker_id"].isna(), "ticker"].unique()
    if len(unmapped):
        print(f"[WARN] Unmapped tickers dropped (not in dim_ticker): {unmapped[:10]}")

    daily_metrics = daily_metrics.dropna(subset=["ticker_id"])
    daily_metrics["ticker_id"] = daily_metrics["ticker_id"].astype(int)
    daily_metrics = daily_metrics.drop(columns=["ticker"])

    rename_dict = {c: metric_map[c] for c in daily_metrics.columns if c in metric_map}
    daily_metrics = daily_metrics.rename(columns=rename_dict)


# -------------------------
# 7) UPSERT prices + metrics into Azure
# -------------------------
PRICE_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


# --- 7a) Upsert prices for FXPair/Futures (price-only, no metrics) ---
if not data_price_only.empty:
    data_price_only = data_price_only.copy()
    data_price_only["ticker_id"] = data_price_only["ticker"].map(ticker_to_id)
    data_price_only = data_price_only.dropna(subset=["ticker_id"])
    data_price_only["ticker_id"] = data_price_only["ticker_id"].astype(int)

    price_cols_available = [c for c in PRICE_COLS if c in data_price_only.columns]
    df_po = data_price_only[["ticker_id", "dt"] + price_cols_available].copy()

    # Volume cleanup
    if "volume" in df_po.columns:
        df_po["volume"] = pd.to_numeric(df_po["volume"], errors="coerce")
        df_po["volume"] = df_po["volume"].apply(
            lambda x: int(x) if x is not None and pd.notna(x) and x < MAX_VALID_VOLUME else None
        )

    cols_sql = ", ".join(["ticker_id", "dt"] + price_cols_available)
    upd_sql = ", ".join([f"{c} = EXCLUDED.{c}" for c in price_cols_available])
    rows = list(df_po.itertuples(index=False, name=None))

    with pg_conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {SCHEMA}.fact_price_daily ({cols_sql})
            VALUES %s
            ON CONFLICT (ticker_id, dt)
            DO UPDATE SET {upd_sql};
            """,
            rows,
            page_size=5000,
        )
    pg_conn.commit()
    print(f"Upserted FX/Futures prices: {len(df_po)}")


# --- 7b) Upsert prices + metrics for Stock/ETF/Index ---
if not daily_metrics.empty:
    df = daily_metrics.copy()
    df = df.rename(columns={"date": "dt"}) if "date" in df.columns and "dt" not in df.columns else df
    df["dt"] = pd.to_datetime(df["dt"]).dt.date

    present_price = [c for c in PRICE_COLS if c in df.columns]
    if present_price:
        df_price = df[["ticker_id", "dt"] + present_price].copy()

        # Null out corrupt volume values — anything above 100 billion is bad FMP data
        if "volume" in df_price.columns:
            df_price["volume"] = pd.to_numeric(df_price["volume"], errors="coerce")
            n_corrupt = int((df_price["volume"] >= MAX_VALID_VOLUME).sum())
            if n_corrupt > 0:
                print(f"[WARN] Nulling {n_corrupt} corrupt volume value(s) exceeding {MAX_VALID_VOLUME:,}")
            df_price["volume"] = df_price["volume"].where(df_price["volume"] < MAX_VALID_VOLUME, other=None)

        # Cast volume from float64 to nullable Python int so psycopg2 sends it as
        # a proper bigint rather than a float, which can overflow on insertion
        if "volume" in df_price.columns:
            df_price["volume"] = df_price["volume"].apply(
                lambda x: int(x) if x is not None and pd.notna(x) else None
            )

        cols_sql = ", ".join(["ticker_id", "dt"] + present_price)
        upd_sql = ", ".join([f"{c} = EXCLUDED.{c}" for c in present_price])
        rows = list(df_price.itertuples(index=False, name=None))

        with pg_conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {SCHEMA}.fact_price_daily ({cols_sql})
                VALUES %s
                ON CONFLICT (ticker_id, dt)
                DO UPDATE SET {upd_sql};
                """,
                rows,
                page_size=5000,
            )
        pg_conn.commit()
        print("Upserted prices:", len(df_price))

    df_no_price = df.drop(columns=[c for c in PRICE_COLS if c in df.columns], errors="ignore")
    NON_METRIC = set(["ticker_id", "dt", "ticker_type"] + PRICE_COLS)
    metric_cols = [c for c in df_no_price.columns if c not in NON_METRIC]

    if metric_cols:
        long_metrics = (
            df_no_price.melt(
                id_vars=["ticker_id", "dt"],
                value_vars=metric_cols,
                var_name="metric_id",
                value_name="value",
            )
            .dropna(subset=["value"])
            .copy()
        )
        long_metrics["metric_id"] = long_metrics["metric_id"].astype(int)

        mrows = list(long_metrics[["ticker_id", "dt", "metric_id", "value"]].itertuples(index=False, name=None))

        with pg_conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {SCHEMA}.fact_metric_daily (ticker_id, dt, metric_id, value)
                VALUES %s
                ON CONFLICT (ticker_id, dt, metric_id)
                DO UPDATE SET value = EXCLUDED.value;
                """,
                mrows,
                page_size=10000,
            )
        pg_conn.commit()
        print("Upserted metrics:", len(long_metrics))
else:
    print("[INFO] No metrics tickers had data — skipping metrics upsert.")


# -------------------------
# 8) Refresh snapshot_metric_latest (NO CREATE TABLE HERE)
# -------------------------
# snapshot_metric_latest MUST exist already.
# Expected schema:
#   (ticker_id INT, metric_id INT, dt DATE, value DOUBLE PRECISION, PRIMARY KEY(ticker_id, metric_id))
#
# This step updates latest dt/value per (ticker_id, metric_id).
with pg_conn.cursor() as cur:
    cur.execute(
        f"""
        INSERT INTO {SCHEMA}.snapshot_metric_latest (ticker_id, metric_id, dt, value)
        SELECT f.ticker_id, f.metric_id, f.dt::date, f.value
        FROM {SCHEMA}.fact_metric_daily f
        JOIN (
          SELECT ticker_id, metric_id, MAX(dt)::date AS dt
          FROM {SCHEMA}.fact_metric_daily
          GROUP BY 1,2
        ) mx
          ON f.ticker_id = mx.ticker_id
         AND f.metric_id = mx.metric_id
         AND f.dt::date = mx.dt
        ON CONFLICT (ticker_id, metric_id)
        DO UPDATE SET dt = EXCLUDED.dt, value = EXCLUDED.value
        WHERE ({SCHEMA}.snapshot_metric_latest.dt IS DISTINCT FROM EXCLUDED.dt)
           OR ({SCHEMA}.snapshot_metric_latest.value IS DISTINCT FROM EXCLUDED.value);
        """
    )
pg_conn.commit()

print("[DONE] Azure updated + snapshot refreshed for", T_DATE)

# -------------------------
# CLEANUP
# -------------------------
try:
    pg_conn.close()
except Exception:
    pass
try:
    engine.dispose()
except Exception:
    pass
