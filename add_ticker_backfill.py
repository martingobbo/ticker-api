# jobs/add_tickers_batch_backfill.py
# =========================
# BATCH ADD TICKERS + BACKFILL (Azure Postgres + FMP)
#
# Runs the full single-ticker backfill pipeline sequentially for a list of tickers
# of the SAME ticker type. Skips failures and continues to the next ticker.
#
# Usage examples:
#   # Inline list
#   python3 jobs/add_tickers_batch_backfill.py \
#     --tickers EURUSD GBPUSD USDJPY AUDUSD \
#     --ticker-type FXPair --start-date 2020-01-01
#
#   # From a file (one ticker per line, # lines are ignored)
#   python3 jobs/add_tickers_batch_backfill.py \
#     --tickers-file jobs/tickers_to_add.txt \
#     --ticker-type Stock --start-date 2020-01-01
#
# TICKER TYPES:
#   Stock    - equity, has adj_close, has volume, profile fetched
#   ETF      - fund, has adj_close, has volume, profile fetched
#   Crypto   - crypto, no adj_close (close=adj_close), has volume, profile fetched
#   Index    - index, no adj_close (close=adj_close), no volume, profile fetched
#   FXPair   - forex, no adj_close (close=adj_close), no volume, no profile
#              FMP symbols: EURUSD, GBPUSD, USDJPY etc (no slash, no =F suffix)
#   Futures  - futures/commodity, no adj_close (close=adj_close), has volume, no profile
#              FMP symbols: GCUSD (Gold), CLUSD (Crude Oil), SIUSD (Silver) etc
#              NOTE: NOT Yahoo Finance format (CL=F). Use FMP commodity symbols.
#
# SECURITY: env vars only
#   FMP_API_KEY, PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, (optional) PGSSLMODE
# =========================

import os
import math
import time
import argparse
import traceback
from datetime import date as _date
from typing import Optional, Dict, Any, List

import requests
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extras import execute_values


# -------------------------
# CLI
# -------------------------
def _parse_args():
    p = argparse.ArgumentParser()

    ticker_source = p.add_mutually_exclusive_group(required=True)
    ticker_source.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Space-separated list of symbols, e.g. EURUSD GBPUSD USDJPY",
    )
    ticker_source.add_argument(
        "--tickers-file",
        metavar="FILE",
        help="Path to a text file with one ticker per line (# lines are ignored)",
    )

    p.add_argument(
        "--ticker-type",
        required=True,
        choices=["Stock", "ETF", "Crypto", "Index", "FXPair", "Futures"],
        help="Ticker type — applies to ALL tickers in this batch",
    )
    p.add_argument("--start-date", default="2020-01-01", help="YYYY-MM-DD")
    p.add_argument("--end-date", default="", help="YYYY-MM-DD (default=today)")
    p.add_argument("--schema", default=os.getenv("SCHEMA", "analytics"))
    p.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between tickers (default 0.5 — be kind to FMP rate limits)",
    )
    p.add_argument(
        "--sleep-profile",
        type=float,
        default=0.0,
        help="Extra sleep between profile and history API calls within a single ticker",
    )
    return p.parse_args()


args = _parse_args()
TICKER_TYPE = args.ticker_type.strip()
SCHEMA = args.schema.strip()
START_DATE = args.start_date
END_DATE = args.end_date.strip() if args.end_date.strip() else _date.today().isoformat()


# -------------------------
# Resolve ticker list
# -------------------------
def _load_tickers_from_file(path: str) -> List[str]:
    tickers = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line.upper())
    return tickers


if args.tickers:
    TICKERS = [t.strip().upper() for t in args.tickers]
else:
    TICKERS = _load_tickers_from_file(args.tickers_file)

if not TICKERS:
    raise SystemExit("[ERROR] No tickers found. Check --tickers or --tickers-file.")

print(f"Batch: {len(TICKERS)} ticker(s) | type={TICKER_TYPE} | {START_DATE}..{END_DATE}")
print(f"  {TICKERS}\n")


# -------------------------
# Type classification
# -------------------------
NO_PROFILE_TYPES  = {"FXPair", "Futures"}
NO_VOLUME_TYPES   = {"FXPair", "Index"}
NO_ADJ_CLOSE_TYPES = {"FXPair", "Futures", "Index", "Crypto"}
USES_VOLUME_TYPES  = {"Stock", "ETF"}


# -------------------------
# ENV / DB setup
# -------------------------
FMP_API_KEY = os.environ["FMP_API_KEY"]
PGHOST      = os.environ["PGHOST"]
PGPORT      = int(os.getenv("PGPORT", "5432"))
PGDATABASE  = os.getenv("PGDATABASE", "postgres")
PGUSER      = os.environ["PGUSER"]
PGPASSWORD  = os.environ["PGPASSWORD"]
PGSSLMODE   = os.getenv("PGSSLMODE", "require")

ENGINE_URL = (
    f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    f"?sslmode={PGSSLMODE}"
)

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "4"))
BACKOFF_BASE    = float(os.getenv("BACKOFF_BASE", "1.6"))

PRICE_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


# -------------------------
# FMP helpers
# -------------------------
def _get_json(url: str, params: Dict[str, Any]) -> Optional[Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE ** attempt)
                continue
            return None

        if r.status_code in (429, 502, 503, 504):
            if attempt < MAX_RETRIES:
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else (BACKOFF_BASE ** attempt)
                time.sleep(delay)
                continue
            return None

        if r.status_code != 200:
            return None

        try:
            return r.json()
        except ValueError:
            return None

    return None


def fetch_profile(symbol: str) -> Dict[str, Any]:
    if TICKER_TYPE in NO_PROFILE_TYPES:
        return {}
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
    js = _get_json(url, {"apikey": FMP_API_KEY})
    if isinstance(js, list) and js:
        return js[0] or {}
    return {}


def fetch_market_cap(symbol: str) -> Optional[float]:
    if TICKER_TYPE in NO_PROFILE_TYPES:
        return None
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
    js = _get_json(url, {"apikey": FMP_API_KEY})
    if isinstance(js, list) and js:
        mc = js[0].get("marketCap")
        try:
            return float(mc) if mc is not None else None
        except Exception:
            return None
    return None


def fetch_history(symbol: str, date_from: str, date_to: str) -> pd.DataFrame:
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    js = _get_json(url, {"from": date_from, "to": date_to, "apikey": FMP_API_KEY})
    if not js or not isinstance(js, dict):
        return pd.DataFrame()

    hist = js.get("historical", [])
    if not hist:
        return pd.DataFrame()

    rows = []
    for h in hist:
        close_val = h.get("close")
        adj_close_val = close_val if TICKER_TYPE in NO_ADJ_CLOSE_TYPES else h.get("adjClose")
        volume_val    = None if TICKER_TYPE in NO_VOLUME_TYPES else h.get("volume")

        rows.append({
            "dt":        h.get("date"),
            "open":      h.get("open"),
            "high":      h.get("high"),
            "low":       h.get("low"),
            "close":     close_val,
            "adj_close": adj_close_val,
            "volume":    volume_val,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    df = df.sort_values("dt").drop_duplicates(subset=["dt"], keep="last").reset_index(drop=True)
    return df


# -------------------------
# Metric calculations
# -------------------------
TRADING_DAYS_PER_YEAR = 252
WIN_05, WIN_10, WIN_15, WIN_20, WIN_50 = 5, 10, 15, 20, 50
WIN_60, WIN_100, WIN_200, WIN_252, WIN_300, WIN_750 = 60, 100, 200, 252, 300, 750
SMA_POS_LEN = 3


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
    return float(np.sum((x - xm) * (y - ym)) / denom)


def max_drawdown_only(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    if not np.isfinite(a).any():
        return np.nan
    peak = np.nan
    start_idx = -1
    for i in range(a.size):
        if math.isfinite(a[i]) and a[i] > 0.0:
            peak = a[i]; start_idx = i; break
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
            max_price = a[i]; max_idx = i; break
    if max_idx < 0:
        return np.nan
    best_dd = 0.0
    best_dur = 0
    for j in range(max_idx + 1, a.size):
        pj = a[j]
        if not math.isfinite(pj) or pj <= 0.0:
            continue
        if pj > max_price:
            max_price = pj; max_idx = j
        dd = pj / max_price - 1.0
        if dd < best_dd:
            best_dd = dd
            best_dur = j - max_idx
    return float(best_dur)


def per_ticker_metrics_all_rows(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["date"] = pd.to_datetime(g["dt"])
    g = g.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    uses_volume = TICKER_TYPE in USES_VOLUME_TYPES

    close     = pd.to_numeric(g["adj_close"], errors="coerce").astype(float)
    open_     = pd.to_numeric(g["open"],      errors="coerce").astype(float)
    high      = pd.to_numeric(g["high"],      errors="coerce").astype(float)
    low       = pd.to_numeric(g["low"],       errors="coerce").astype(float)
    vol_orig  = pd.to_numeric(g["volume"],    errors="coerce").astype(float)
    vol_calc  = vol_orig if uses_volume else pd.Series(np.nan, index=g.index)

    close_safe   = close.replace(0, np.nan)
    log_price    = np.log(close_safe)
    prev_close   = close_safe.shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.where(
            (close_safe > 0) & (prev_close > 0), np.log(close_safe / prev_close), np.nan
        )
    log_returns = pd.Series(log_returns, index=g.index)
    if len(g) > 0 and pd.isna(log_returns.iloc[0]):
        log_returns.iloc[0] = 0.0

    ma20  = close.rolling(WIN_20,  min_periods=1).mean()
    ma50  = close.rolling(WIN_50,  min_periods=1).mean()
    ma100 = close.rolling(WIN_100, min_periods=1).mean()
    ma200 = close.rolling(WIN_200, min_periods=1).mean()

    dv = close * vol_calc
    sum_px_vol_20 = dv.rolling(WIN_20, min_periods=1).sum()
    sum_vol_20    = vol_calc.rolling(WIN_20, min_periods=1).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap20 = np.where(sum_vol_20 > 0, sum_px_vol_20 / sum_vol_20, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_dv = np.where((vol_calc > 0) & (close > 0), np.log(vol_calc * close), np.nan)
    ln_dv = pd.Series(ln_dv, index=g.index)
    vol_accel_5d  = ln_dv - ln_dv.shift(5)
    vol_accel_10d = ln_dv - ln_dv.shift(10)

    avg10_dv = dv.rolling(WIN_10, min_periods=1).mean()
    avg60_dv = dv.rolling(WIN_60, min_periods=1).mean()
    std60_dv = dv.rolling(WIN_60, min_periods=2).std(ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        abn_vol_60d = np.where(std60_dv > 0, (avg10_dv - avg60_dv) / std60_dv, np.nan)

    vol20_ann  = log_returns.rolling(WIN_20,  min_periods=2).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol100_ann = log_returns.rolling(WIN_100, min_periods=2).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    mean100    = log_returns.rolling(WIN_100, min_periods=1).mean()

    low_10  = low.rolling(WIN_10,  min_periods=1).min()
    high_10 = high.rolling(WIN_10, min_periods=1).max()
    rng_10  = high_10 - low_10
    with np.errstate(invalid="ignore", divide="ignore"):
        pos_10d = np.where(rng_10 != 0, (close - low_10) / rng_10, 0.0)
    five_day_range_pos = pd.Series(pos_10d, index=g.index).rolling(SMA_POS_LEN, min_periods=1).mean()

    daily_range = high - low
    avg_rng_10  = daily_range.rolling(WIN_10, min_periods=1).mean()
    avg_rng_60  = daily_range.rolling(WIN_60, min_periods=1).mean()
    std_rng_60  = daily_range.rolling(WIN_60, min_periods=2).std(ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_60_10_highlowrange = np.where(std_rng_60 > 0, (avg_rng_10 - avg_rng_60) / std_rng_60, 0.0)

    def safe_lr(curr, lagged):
        return np.where((curr > 0) & (lagged > 0), np.log(curr / lagged), 0.0)

    ret5   = safe_lr(close, close.shift(5))
    ret10  = safe_lr(close, close.shift(10))
    ret20  = safe_lr(close, close.shift(20))
    ret40  = safe_lr(close, close.shift(40))
    ret60  = safe_lr(close, close.shift(60))
    ret200 = safe_lr(close, close.shift(200))
    ret300 = safe_lr(close, close.shift(300))

    median_return_100d = log_returns.rolling(WIN_100, min_periods=1).median()

    closes_by_date = pd.Series(close.values, index=g["date"])

    def _dd_percent(w):
        v = pd.to_numeric(w, errors="coerce").to_numpy(dtype="float64")
        if v.size <= 1 or not np.isfinite(v).any(): return 0.0
        try: peak_pos = np.nanargmax(v)
        except ValueError: return 0.0
        peak = v[peak_pos]
        if not np.isfinite(peak): return 0.0
        suffix = v[peak_pos:]
        if suffix.size == 0 or not np.isfinite(suffix).any(): return 0.0
        return float(np.nanmin(suffix) / peak - 1.0)

    def _dd_duration(w):
        v = pd.to_numeric(w, errors="coerce").to_numpy(dtype="float64")
        idx = w.index
        if v.size <= 1 or not np.isfinite(v).any(): return 0.0
        try: peak_pos = np.nanargmax(v)
        except ValueError: return 0.0
        suffix = v[peak_pos:]
        if suffix.size == 0 or not np.isfinite(suffix).any(): return 0.0
        trough_pos = peak_pos + np.nanargmin(suffix)
        return float((idx[trough_pos] - idx[peak_pos]).days)

    drawdown_percent_100 = closes_by_date.rolling(100, min_periods=1).apply(_dd_percent, raw=False).values
    drawdown_days_100    = closes_by_date.rolling(100, min_periods=1).apply(_dd_duration, raw=False).values

    lr      = pd.Series(log_returns, index=g.index)
    vol_5   = lr.rolling(WIN_05,  min_periods=WIN_05).std(ddof=1)
    vol_15  = lr.rolling(WIN_15,  min_periods=WIN_15).std(ddof=1)
    vol_60  = lr.rolling(WIN_60,  min_periods=WIN_60).std(ddof=1)
    vol_252 = lr.rolling(WIN_252, min_periods=WIN_252).std(ddof=1)

    neg   = np.minimum(lr, 0.0)
    pos_r = np.maximum(lr, 0.0)
    dd_15  = (pd.Series(neg).pow(2).rolling(WIN_15,  min_periods=WIN_15).mean())  ** 0.5
    dd_60  = (pd.Series(neg).pow(2).rolling(WIN_60,  min_periods=WIN_60).mean())  ** 0.5
    dd_252 = (pd.Series(neg).pow(2).rolling(WIN_252, min_periods=WIN_252).mean()) ** 0.5
    ud_15  = (pd.Series(pos_r).pow(2).rolling(WIN_15,  min_periods=WIN_15).mean())  ** 0.5
    ud_60  = (pd.Series(pos_r).pow(2).rolling(WIN_60,  min_periods=WIN_60).mean())  ** 0.5
    ud_252 = (pd.Series(pos_r).pow(2).rolling(WIN_252, min_periods=WIN_252).mean()) ** 0.5

    hl_log = np.log(high.replace(0, np.nan) / low.replace(0, np.nan))
    k      = 1.0 / (4.0 * math.log(2.0))
    pk20   = np.sqrt(k * hl_log.pow(2).rolling(WIN_20, min_periods=WIN_20).mean())

    sum10         = lr.rolling(WIN_10, min_periods=WIN_10).sum()
    change_10dayret = sum10 - sum10.shift(WIN_10)

    slope_lp_recent60 = (
        pd.Series(np.log(close.replace(0, np.nan)))
        .rolling(WIN_60, min_periods=WIN_60)
        .apply(ols_slope_window, raw=True)
    )
    slope_lp_prev60 = slope_lp_recent60.shift(WIN_60)
    ret_accel_60    = slope_lp_recent60 - slope_lp_prev60

    slope_vol60_over20  = vol_60.rolling(WIN_20,  min_periods=WIN_20).apply(ols_slope_window, raw=True)
    slope_vol252_over60 = vol_252.rolling(WIN_60, min_periods=WIN_60).apply(ols_slope_window, raw=True)

    dv_sma_252    = dv.rolling(WIN_252, min_periods=WIN_252).mean()
    dv_sma_60     = dv.rolling(WIN_60,  min_periods=WIN_60).mean()
    dv252_accel_60 = dv_sma_252.rolling(WIN_60, min_periods=WIN_60).apply(ols_slope_window, raw=True)
    corr_px_dv_60  = close.rolling(WIN_60, min_periods=WIN_60).corr(dv)

    ema5_of_vol15 = vol_15.ewm(span=5, adjust=False).mean()
    mdd_750       = close.rolling(WIN_750, min_periods=2).apply(max_drawdown_only,          raw=True)
    mdd_dur_750   = close.rolling(WIN_750, min_periods=2).apply(max_drawdown_duration_only, raw=True)

    out = pd.DataFrame({
        "dt":        g["date"].dt.date.values,
        "open":      open_.values,
        "high":      high.values,
        "low":       low.values,
        "close":     pd.to_numeric(g["close"], errors="coerce").astype(float).values,
        "adj_close": close.values,
        "volume":    vol_orig.values,
        "log_returns":                          log_returns.values,
        "volatility_20d":                       vol20_ann.values,
        "volatility_100d":                      vol100_ann.values,
        "mean_return_100d":                     mean100.values,
        "moving_avg_20d":                       ma20.values,
        "moving_avg_50d":                       ma50.values,
        "moving_avg_100d":                      ma100.values,
        "moving_avg_200d":                      ma200.values,
        "vwap_20d":                             vwap20,
        "vol_accel_5d":                         vol_accel_5d.values,
        "vol_accel_10d":                        vol_accel_10d.values,
        "abn_vol_60d":                          abn_vol_60d,
        "5_day_range_pos":                      five_day_range_pos.values,
        "60_10_highlowrange_zscore":            z_60_10_highlowrange,
        "5_day_ret":                            ret5,
        "10_day_ret":                           ret10,
        "20_day_ret":                           ret20,
        "40_day_ret":                           ret40,
        "60_day_ret":                           ret60,
        "200_day_ret":                          ret200,
        "300_day_ret":                          ret300,
        "median_return_100d":                   median_return_100d.values,
        "drawdown_percent":                     drawdown_percent_100,
        "drawdown_duration_days":               drawdown_days_100,
        "log_prices":                           log_price.values,
        "change_10dayret":                      change_10dayret.values,
        "slope_over60_of_logprice":             slope_lp_recent60.values,
        "prior_slope_over60_of_logprice":       slope_lp_prev60.values,
        "60d_return_accel":                     ret_accel_60.values,
        "750d_drawdown":                        mdd_750.values,
        "750d_drawdownduration":                mdd_dur_750.values,
        "15d_downsidedeviation":                dd_15.values,
        "60d_downsidedeviation":                dd_60.values,
        "252d_downsidedeviation":               dd_252.values,
        "15d_upsidevolatility":                 ud_15.values,
        "60d_upsidevolatility":                 ud_60.values,
        "252d_upsidevolatility":                ud_252.values,
        "5d_volatility":                        vol_5.values,
        "15d_volatility":                       vol_15.values,
        "60d_volatility":                       vol_60.values,
        "252d_volatility":                      vol_252.values,
        "20d_parkinson_HL_volatility":          pk20.values,
        "5d_EMA_15dayvolatility":               ema5_of_vol15.values,
        "slope_over20_of_60d_volatility":       slope_vol60_over20.values,
        "slope_over60_of_252d_volatility":      slope_vol252_over60.values,
        "252d_dollar_volume_SMA":               dv_sma_252.values,
        "60d_dollar_volume_SMA":                dv_sma_60.values,
        "252d_dollar_volume_accel":             dv252_accel_60.values,
        "60d_price_dollarVolume_correlation":   corr_px_dv_60.values,
    })

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out[["5_day_range_pos", "60_10_highlowrange_zscore", "drawdown_percent", "drawdown_duration_days"]] = (
        out[["5_day_range_pos", "60_10_highlowrange_zscore", "drawdown_percent", "drawdown_duration_days"]]
        .fillna(0.0)
    )
    return out


def _to_int(x):
    try: return int(float(x))
    except Exception: return None

def _to_float(x):
    try: return float(x)
    except Exception: return None


# -------------------------
# Single-ticker pipeline
# -------------------------
def run_one(symbol: str, engine, pg_conn) -> bool:
    """
    Runs the full 5-step backfill for one ticker.
    Returns True on success, False on any failure.
    """
    print(f"\n{'='*60}")
    print(f"  {symbol}  ({TICKER_TYPE})  {START_DATE}..{END_DATE}")
    print(f"{'='*60}")

    # --- Step 1: dim_ticker upsert ---
    print(f"  [1/5] Upserting dim_ticker ...")
    profile = fetch_profile(symbol)
    if args.sleep_profile:
        time.sleep(args.sleep_profile)
    mcap = fetch_market_cap(symbol)

    row = {
        "ticker":       symbol,
        "name":         profile.get("companyName") or profile.get("name") or None,
        "market_cap":   _to_int(mcap),
        "beta":         _to_float(profile.get("beta")),
        "employees":    _to_int(profile.get("fullTimeEmployees") or profile.get("employees")),
        "industry":     profile.get("industry"),
        "headquarters": profile.get("address") or profile.get("headQuarter") or profile.get("headquarters"),
        "exchange":     profile.get("exchangeShortName") or profile.get("exchange"),
        "gics_sector":  profile.get("sector"),
        "gics_subsector": profile.get("industry"),
        "ticker_type":  TICKER_TYPE,
    }

    with pg_conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {SCHEMA}.dim_ticker
              (ticker, name, market_cap, beta, employees, industry, headquarters,
               exchange, gics_sector, gics_subsector, ticker_type)
            VALUES
              (%(ticker)s, %(name)s, %(market_cap)s, %(beta)s, %(employees)s,
               %(industry)s, %(headquarters)s, %(exchange)s, %(gics_sector)s,
               %(gics_subsector)s, %(ticker_type)s)
            ON CONFLICT (ticker)
            DO UPDATE SET
              name           = COALESCE(EXCLUDED.name,           {SCHEMA}.dim_ticker.name),
              market_cap     = COALESCE(EXCLUDED.market_cap,     {SCHEMA}.dim_ticker.market_cap),
              beta           = COALESCE(EXCLUDED.beta,           {SCHEMA}.dim_ticker.beta),
              employees      = COALESCE(EXCLUDED.employees,      {SCHEMA}.dim_ticker.employees),
              industry       = COALESCE(EXCLUDED.industry,       {SCHEMA}.dim_ticker.industry),
              headquarters   = COALESCE(EXCLUDED.headquarters,   {SCHEMA}.dim_ticker.headquarters),
              exchange       = COALESCE(EXCLUDED.exchange,       {SCHEMA}.dim_ticker.exchange),
              gics_sector    = COALESCE(EXCLUDED.gics_sector,    {SCHEMA}.dim_ticker.gics_sector),
              gics_subsector = COALESCE(EXCLUDED.gics_subsector, {SCHEMA}.dim_ticker.gics_subsector),
              ticker_type    = EXCLUDED.ticker_type
            RETURNING ticker_id;
            """,
            row,
        )
        ticker_id = cur.fetchone()[0]
    pg_conn.commit()
    print(f"     ticker_id = {ticker_id}")

    # --- Step 2: Download history ---
    print(f"  [2/5] Downloading history from FMP ...")
    hist = fetch_history(symbol, START_DATE, END_DATE)
    if hist.empty:
        print(f"  [SKIP] No history returned for {symbol}.")
        print(f"         FX: use EURUSD format. Futures: use GCUSD format (not GC=F).")
        return False

    hist["ticker_id"] = int(ticker_id)
    hist = hist[["ticker_id", "dt", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    hist = hist.sort_values("dt").reset_index(drop=True)
    print(f"     {len(hist):,} rows  ({hist['dt'].min()} .. {hist['dt'].max()})")

    # --- Step 3: Upsert prices ---
    print(f"  [3/5] Upserting prices ...")
    price_rows = list(hist.itertuples(index=False, name=None))
    with pg_conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {SCHEMA}.fact_price_daily
              (ticker_id, dt, open, high, low, close, adj_close, volume)
            VALUES %s
            ON CONFLICT (ticker_id, dt)
            DO UPDATE SET
              open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
              close=EXCLUDED.close, adj_close=EXCLUDED.adj_close, volume=EXCLUDED.volume;
            """,
            price_rows,
            page_size=5000,
        )
    pg_conn.commit()
    print(f"     upserted {len(hist):,} price rows")

    # --- Step 3.5: Reload from DB (source of truth) ---
    prices_db = pd.read_sql(
        text(f"""
            SELECT dt, open, high, low, close, adj_close, volume
            FROM {SCHEMA}.fact_price_daily
            WHERE ticker_id = :tid
            ORDER BY dt
        """),
        engine,
        params={"tid": int(ticker_id)},
    )
    if prices_db.empty:
        print(f"  [SKIP] No prices in DB after upsert for {symbol}.")
        return False

    prices_db["ticker_id"] = int(ticker_id)
    prices_db = prices_db[["ticker_id", "dt", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    prices_db["dt"] = pd.to_datetime(prices_db["dt"]).dt.date

    # --- Step 4: Compute + upsert metrics ---
    print(f"  [4/5] Computing + upserting metrics ...")
    metrics_wide = per_ticker_metrics_all_rows(prices_db)
    metrics_wide["ticker_id"] = int(ticker_id)

    dim_metric = pd.read_sql(
        text(f"SELECT metric_id, metric_code FROM {SCHEMA}.dim_metric"), engine
    )
    metric_map = dict(zip(dim_metric["metric_code"], dim_metric["metric_id"]))

    non_metric_cols = {"ticker_id", "dt"} | set(PRICE_COLS)
    metric_cols = [c for c in metrics_wide.columns if c not in non_metric_cols and c in metric_map]

    missing = [c for c in metrics_wide.columns if c not in non_metric_cols and c not in metric_map]
    if missing:
        print(f"     [WARN] {len(missing)} computed cols not in dim_metric (skipped)")

    if not metric_cols:
        print(f"  [SKIP] No metric columns matched dim_metric for {symbol}.")
        return False

    long_metrics = (
        metrics_wide[["ticker_id", "dt"] + metric_cols]
        .melt(id_vars=["ticker_id", "dt"], value_vars=metric_cols,
              var_name="metric_code", value_name="value")
        .dropna(subset=["value"])
        .copy()
    )
    long_metrics["metric_id"] = long_metrics["metric_code"].map(metric_map).astype(int)
    long_metrics = long_metrics[["ticker_id", "dt", "metric_id", "value"]]

    with pg_conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {SCHEMA}.fact_metric_daily (ticker_id, dt, metric_id, value)
            VALUES %s
            ON CONFLICT (ticker_id, dt, metric_id)
            DO UPDATE SET value = EXCLUDED.value;
            """,
            list(long_metrics.itertuples(index=False, name=None)),
            page_size=10000,
        )
    pg_conn.commit()
    print(f"     upserted {len(long_metrics):,} metric rows ({len(metric_cols)} metric cols)")

    # --- Step 5: Refresh snapshot ---
    print(f"  [5/5] Refreshing snapshot_metric_latest ...")
    with pg_conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {SCHEMA}.snapshot_metric_latest (ticker_id, metric_id, dt, value)
            SELECT f.ticker_id, f.metric_id, f.dt::date, f.value
            FROM {SCHEMA}.fact_metric_daily f
            JOIN (
              SELECT ticker_id, metric_id, MAX(dt)::date AS dt
              FROM {SCHEMA}.fact_metric_daily
              WHERE ticker_id = %s
              GROUP BY 1,2
            ) mx
              ON f.ticker_id = mx.ticker_id
             AND f.metric_id = mx.metric_id
             AND f.dt::date  = mx.dt
            WHERE f.ticker_id = %s
            ON CONFLICT (ticker_id, metric_id)
            DO UPDATE SET dt = EXCLUDED.dt, value = EXCLUDED.value
            WHERE ({SCHEMA}.snapshot_metric_latest.dt    IS DISTINCT FROM EXCLUDED.dt)
               OR ({SCHEMA}.snapshot_metric_latest.value IS DISTINCT FROM EXCLUDED.value);
            """,
            (int(ticker_id), int(ticker_id)),
        )
    pg_conn.commit()

    print(f"  [DONE] {symbol} ✓")
    return True


# -------------------------
# Main batch loop
# -------------------------
engine  = create_engine(ENGINE_URL, pool_pre_ping=True)
pg_conn = psycopg2.connect(
    host=PGHOST, port=PGPORT, dbname=PGDATABASE,
    user=PGUSER, password=PGPASSWORD, sslmode=PGSSLMODE,
)

succeeded = []
failed    = []

for i, symbol in enumerate(TICKERS):
    try:
        ok = run_one(symbol, engine, pg_conn)
        if ok:
            succeeded.append(symbol)
        else:
            failed.append((symbol, "skipped — no data or no metric match"))
    except Exception as e:
        pg_conn.rollback()  # don't let one failure poison the connection
        failed.append((symbol, str(e)))
        print(f"\n  [ERROR] {symbol} failed: {e}")
        traceback.print_exc()

    # Sleep between tickers (not after the last one)
    if i < len(TICKERS) - 1 and args.sleep > 0:
        print(f"\n  (sleeping {args.sleep}s before next ticker...)")
        time.sleep(args.sleep)

# -------------------------
# Summary
# -------------------------
print(f"\n{'='*60}")
print(f"BATCH COMPLETE  —  {len(TICKERS)} tickers processed")
print(f"  ✓ succeeded : {len(succeeded)}  {succeeded}")
if failed:
    print(f"  ✗ failed    : {len(failed)}")
    for sym, reason in failed:
        print(f"      {sym}: {reason}")
print(f"{'='*60}")

try:
    pg_conn.close()
except Exception:
    pass
try:
    engine.dispose()
except Exception:
    pass
