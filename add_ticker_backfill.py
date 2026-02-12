# jobs/add_ticker_backfill.py
# =========================
# ADD TICKER + BACKFILL (Azure Postgres + FMP)
#
# - Upserts a row into analytics.dim_ticker for a new symbol (ticker_id is IDENTITY)
# - Pulls OHLCV history from FMP from START_DATE..END_DATE
# - Upserts prices into analytics.fact_price_daily
# - Reloads price history for that ticker from Azure (source of truth)
# - Computes metrics for ALL dates and upserts into analytics.fact_metric_daily
# - Refreshes analytics.snapshot_metric_latest for THAT ticker
#
# SECURITY: env vars only
#   FMP_API_KEY, PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, (optional) PGSSLMODE
# =========================

import os
import math
import time
import argparse
from datetime import date as _date
from typing import Optional, Dict, Any

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
    p.add_argument("--ticker", required=True, help="Symbol, e.g. AAPL")
    p.add_argument("--ticker-type", required=True, choices=["Stock", "ETF", "Crypto", "Index"], help="Ticker type")
    p.add_argument("--start-date", default="2020-01-01", help="YYYY-MM-DD")
    p.add_argument("--end-date", default="", help="YYYY-MM-DD (default=today)")
    p.add_argument("--schema", default=os.getenv("SCHEMA", "analytics"))
    p.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between API calls")
    return p.parse_args()


args = _parse_args()
SYMBOL = args.ticker.strip().upper()
TICKER_TYPE = args.ticker_type.strip()
SCHEMA = args.schema.strip()

START_DATE = args.start_date
END_DATE = args.end_date.strip() if args.end_date.strip() else _date.today().isoformat()

# -------------------------
# ENV VARS ONLY
# -------------------------
FMP_API_KEY = os.environ["FMP_API_KEY"]

PGHOST = os.environ["PGHOST"]
PGPORT = 5432
PGDATABASE = os.getenv("PGDATABASE", "postgres")
PGUSER = os.environ["PGUSER"]
PGPASSWORD = os.environ["PGPASSWORD"]
PGSSLMODE = os.getenv("PGSSLMODE", "require")

from sqlalchemy.engine import URL as SA_URL
ENGINE_URL = SA_URL.create(
    drivername="postgresql+psycopg2",
    username=PGUSER,
    password=PGPASSWORD,
    host=PGHOST,
    port=5432,
    database=PGDATABASE,
    query={"sslmode": PGSSLMODE},
)
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

engine = create_engine(ENGINE_URL, pool_pre_ping=True)

pg_conn = psycopg2.connect(
    host=PGHOST,
    port=PGPORT,
    dbname=PGDATABASE,
    user=PGUSER,
    password=PGPASSWORD,
    sslmode=PGSSLMODE,
)

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1.6"))

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
            return r.json()
        except ValueError:
            return None

    return None


def fetch_profile(symbol: str) -> Dict[str, Any]:
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
    js = _get_json(url, {"apikey": FMP_API_KEY})
    if isinstance(js, list) and js:
        return js[0] or {}
    return {}


def fetch_market_cap(symbol: str) -> Optional[float]:
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
        rows.append(
            {
                "dt": h.get("date"),
                "open": h.get("open"),
                "high": h.get("high"),
                "low": h.get("low"),
                "close": h.get("close"),
                "adj_close": h.get("adjClose"),
                "volume": h.get("volume"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    df = df.sort_values("dt").drop_duplicates(subset=["dt"], keep="last").reset_index(drop=True)
    return df


# -------------------------
# Metric calc (same logic as your daily runner, but returns ALL rows)
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


def per_ticker_metrics_all_rows(g: pd.DataFrame, ticker_type: str) -> pd.DataFrame:
    g = g.copy()
    g["date"] = pd.to_datetime(g["dt"])
    g = g.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    ttype = ticker_type.strip().lower()
    uses_volume = (ttype == "stock")

    close = pd.to_numeric(g["adj_close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(g["open"], errors="coerce").astype(float)
    high = pd.to_numeric(g["high"], errors="coerce").astype(float)
    low = pd.to_numeric(g["low"], errors="coerce").astype(float)

    vol_original = pd.to_numeric(g["volume"], errors="coerce").astype(float)
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
            "dt": g["date"].dt.date.values,
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


def _to_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


# -------------------------
# 1) UPSERT dim_ticker (RETURNING ticker_id)
# -------------------------
print(f"[1/5] Upserting dim_ticker for {SYMBOL} ...")

profile = fetch_profile(SYMBOL)
if args.sleep:
    time.sleep(args.sleep)
mcap = fetch_market_cap(SYMBOL)

row = {
    "ticker": SYMBOL,
    "name": profile.get("companyName") or profile.get("name") or None,
    "market_cap": _to_int(mcap),
    "beta": _to_float(profile.get("beta")),
    "employees": _to_int(profile.get("fullTimeEmployees") or profile.get("employees")),
    "industry": profile.get("industry"),
    "headquarters": profile.get("address") or profile.get("headQuarter") or profile.get("headquarters"),
    "exchange": profile.get("exchangeShortName") or profile.get("exchange"),
    "gics_sector": profile.get("sector"),
    "gics_subsector": profile.get("industry"),
    "ticker_type": TICKER_TYPE,
}

with pg_conn.cursor() as cur:
    cur.execute(
        f"""
        INSERT INTO {SCHEMA}.dim_ticker
          (ticker, name, market_cap, beta, employees, industry, headquarters, exchange, gics_sector, gics_subsector, ticker_type)
        VALUES
          (%(ticker)s, %(name)s, %(market_cap)s, %(beta)s, %(employees)s, %(industry)s, %(headquarters)s, %(exchange)s, %(gics_sector)s, %(gics_subsector)s, %(ticker_type)s)
        ON CONFLICT (ticker)
        DO UPDATE SET
          name           = COALESCE(EXCLUDED.name, {SCHEMA}.dim_ticker.name),
          market_cap     = COALESCE(EXCLUDED.market_cap, {SCHEMA}.dim_ticker.market_cap),
          beta           = COALESCE(EXCLUDED.beta, {SCHEMA}.dim_ticker.beta),
          employees      = COALESCE(EXCLUDED.employees, {SCHEMA}.dim_ticker.employees),
          industry       = COALESCE(EXCLUDED.industry, {SCHEMA}.dim_ticker.industry),
          headquarters   = COALESCE(EXCLUDED.headquarters, {SCHEMA}.dim_ticker.headquarters),
          exchange       = COALESCE(EXCLUDED.exchange, {SCHEMA}.dim_ticker.exchange),
          gics_sector    = COALESCE(EXCLUDED.gics_sector, {SCHEMA}.dim_ticker.gics_sector),
          gics_subsector = COALESCE(EXCLUDED.gics_subsector, {SCHEMA}.dim_ticker.gics_subsector),
          ticker_type    = EXCLUDED.ticker_type
        RETURNING ticker_id;
        """,
        row,
    )
    ticker_id = cur.fetchone()[0]
pg_conn.commit()

print(f"  ticker_id = {ticker_id}")


# -------------------------
# 2) DOWNLOAD FULL HISTORY (FMP)
# -------------------------
print(f"[2/5] Downloading history {START_DATE}..{END_DATE} for {SYMBOL} ...")
hist = fetch_history(SYMBOL, START_DATE, END_DATE)
if hist.empty:
    print(f"[STOP] No history returned for {SYMBOL} in {START_DATE}..{END_DATE}. Exiting.")
    raise SystemExit(0)

hist["ticker_id"] = int(ticker_id)
hist = hist[["ticker_id", "dt", "open", "high", "low", "close", "adj_close", "volume"]].copy()
hist = hist.sort_values("dt").reset_index(drop=True)

print(f"  rows fetched: {len(hist):,} ({hist['dt'].min()} .. {hist['dt'].max()})")


# -------------------------
# 3) UPSERT PRICES
# -------------------------
print("[3/5] Upserting prices into fact_price_daily ...")

rows = list(hist.itertuples(index=False, name=None))
cols_sql = "ticker_id, dt, open, high, low, close, adj_close, volume"
upd_sql = "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, adj_close=EXCLUDED.adj_close, volume=EXCLUDED.volume"

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

print(f"  upserted price rows: {len(hist):,}")


# -------------------------
# 3.5) RELOAD FULL HISTORY FROM AZURE (source of truth)
# -------------------------
prices_db = pd.read_sql(
    text(
        f"""
        SELECT dt, open, high, low, close, adj_close, volume
        FROM {SCHEMA}.fact_price_daily
        WHERE ticker_id = :tid
        ORDER BY dt
        """
    ),
    engine,
    params={"tid": int(ticker_id)},
)
if prices_db.empty:
    print("[STOP] No prices in fact_price_daily after upsert. Exiting.")
    raise SystemExit(0)

prices_db["ticker_id"] = int(ticker_id)
prices_db = prices_db[["ticker_id", "dt", "open", "high", "low", "close", "adj_close", "volume"]].copy()
prices_db["dt"] = pd.to_datetime(prices_db["dt"]).dt.date


# -------------------------
# 4) COMPUTE + UPSERT METRICS (ALL DATES)
# -------------------------
print("[4/5] Computing metrics for ALL dates and upserting into fact_metric_daily ...")

metrics_wide = per_ticker_metrics_all_rows(prices_db.rename(columns={"dt": "dt"}), ticker_type=TICKER_TYPE)
metrics_wide["ticker_id"] = int(ticker_id)

dim_metric = pd.read_sql(text(f"SELECT metric_id, metric_code FROM {SCHEMA}.dim_metric"), engine)
metric_map = dict(zip(dim_metric["metric_code"], dim_metric["metric_id"]))

non_metric_cols = {"ticker_id", "dt"} | set(PRICE_COLS)
metric_cols = [c for c in metrics_wide.columns if c not in non_metric_cols and c in metric_map]

missing_in_dim_metric = [c for c in metrics_wide.columns if c not in non_metric_cols and c not in metric_map]
if missing_in_dim_metric:
    print("  [WARN] Computed columns NOT in dim_metric (skipped):")
    print("   ", ", ".join(missing_in_dim_metric[:60]) + (" ..." if len(missing_in_dim_metric) > 60 else ""))

if not metric_cols:
    print("[STOP] No metric columns matched dim_metric. Exiting without metric writes.")
    raise SystemExit(0)

long_metrics = (
    metrics_wide[["ticker_id", "dt"] + metric_cols]
    .melt(id_vars=["ticker_id", "dt"], value_vars=metric_cols, var_name="metric_code", value_name="value")
    .dropna(subset=["value"])
    .copy()
)

long_metrics["metric_id"] = long_metrics["metric_code"].map(metric_map).astype(int)
long_metrics = long_metrics[["ticker_id", "dt", "metric_id", "value"]]

mrows = list(long_metrics.itertuples(index=False, name=None))

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

print(f"  upserted metric rows: {len(long_metrics):,} (wide cols matched: {len(metric_cols)})")


# -------------------------
# 5) REFRESH snapshot_metric_latest (ONLY THIS TICKER)
# -------------------------
print("[5/5] Refreshing snapshot_metric_latest for this ticker ...")

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
         AND f.dt::date = mx.dt
        WHERE f.ticker_id = %s
        ON CONFLICT (ticker_id, metric_id)
        DO UPDATE SET dt = EXCLUDED.dt, value = EXCLUDED.value
        WHERE ({SCHEMA}.snapshot_metric_latest.dt IS DISTINCT FROM EXCLUDED.dt)
           OR ({SCHEMA}.snapshot_metric_latest.value IS DISTINCT FROM EXCLUDED.value);
        """,
        (int(ticker_id), int(ticker_id)),
    )
pg_conn.commit()

print(f"[DONE] {SYMBOL} added + backfilled ({START_DATE}..{END_DATE}).")


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
