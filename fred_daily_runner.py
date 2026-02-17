# jobs/add_fred_backfill.py
# =========================
# ADD FRED MACRO TICKER + BACKFILL (Azure Postgres + FRED API)
#
# - Upserts a row into analytics.dim_ticker for a FRED macro series
#   (ticker_type = 'Macro' by convention)
# - Pulls observation history from FRED API (START_DATE..END_DATE)
# - Stores into analytics.fact_price_daily using:
#     close = adj_close = observed value
#     open / high / low / volume = NULL  (FRED = single value per date)
# - Refreshes analytics.snapshot_metric_latest for THAT ticker
#
# FRED SERIES REGISTRY
#   Pass --ticker with a key from FRED_SERIES below, OR any raw FRED series ID.
#
# SECURITY: env vars only
#   FRED_API_KEY, PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, (optional) PGSSLMODE
# =========================

import os
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


FRED_SERIES: Dict[str, Dict[str, str]] = {

    # ── Policy rates ──────────────────────────────────────────────────────────
    "DFF": {
        "fred_id":   "DFF",
        "name":      "Federal Funds Effective Rate (Daily)",
        "sector":    "Policy Rates",
        "subsector": "Short-Term Policy Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "FEDFUNDS": {
        "fred_id":   "FEDFUNDS",
        "name":      "Federal Funds Effective Rate (Monthly)",
        "sector":    "Policy Rates",
        "subsector": "Short-Term Policy Rate",
        "exchange":  "FRED",
        "frequency": "monthly",
    },
    "SOFR": {
        "fred_id":   "SOFR",
        "name":      "Secured Overnight Financing Rate",
        "sector":    "Policy Rates",
        "subsector": "Overnight Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DFEDTARL": {
        "fred_id":   "DFEDTARL",
        "name":      "Federal Funds Target Rate - Lower Bound",
        "sector":    "Policy Rates",
        "subsector": "Target Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DFEDTARU": {
        "fred_id":   "DFEDTARU",
        "name":      "Federal Funds Target Rate - Upper Bound",
        "sector":    "Policy Rates",
        "subsector": "Target Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },

    # ── Treasury yields ───────────────────────────────────────────────────────
    "DGS1MO": {
        "fred_id":   "DGS1MO",
        "name":      "1-Month Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Short End",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS3MO": {
        "fred_id":   "DGS3MO",
        "name":      "3-Month Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Short End",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS6MO": {
        "fred_id":   "DGS6MO",
        "name":      "6-Month Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Short End",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS1": {
        "fred_id":   "DGS1",
        "name":      "1-Year Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Short End",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS2": {
        "fred_id":   "DGS2",
        "name":      "2-Year Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Mid Curve",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS3": {
        "fred_id":   "DGS3",
        "name":      "3-Year Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Mid Curve",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS5": {
        "fred_id":   "DGS5",
        "name":      "5-Year Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Mid Curve",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS7": {
        "fred_id":   "DGS7",
        "name":      "7-Year Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Long End",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS10": {
        "fred_id":   "DGS10",
        "name":      "10-Year Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Long End",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS20": {
        "fred_id":   "DGS20",
        "name":      "20-Year Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Long End",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DGS30": {
        "fred_id":   "DGS30",
        "name":      "30-Year Treasury Constant Maturity Rate",
        "sector":    "Treasury Yields",
        "subsector": "Long End",
        "exchange":  "FRED",
        "frequency": "daily",
    },

    # ── Yield curve spreads (pre-computed by FRED) ────────────────────────────
    "T10Y2Y": {
        "fred_id":   "T10Y2Y",
        "name":      "10-Year minus 2-Year Treasury Spread",
        "sector":    "Yield Curve Spreads",
        "subsector": "10Y-2Y Spread",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "T10Y3M": {
        "fred_id":   "T10Y3M",
        "name":      "10-Year minus 3-Month Treasury Spread",
        "sector":    "Yield Curve Spreads",
        "subsector": "10Y-3M Spread",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "T5Y5Y": {
        "fred_id":   "T5YIFR",
        "name":      "5-Year, 5-Year Forward Inflation Expectation Rate",
        "sector":    "Yield Curve Spreads",
        "subsector": "Forward Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "T10YFF": {
        "fred_id":   "T10YFF",
        "name":      "10-Year Treasury minus Fed Funds Rate",
        "sector":    "Yield Curve Spreads",
        "subsector": "Policy Spread",
        "exchange":  "FRED",
        "frequency": "daily",
    },

    # ── Real rates & TIPS / Breakevens ────────────────────────────────────────
    "DFII5": {
        "fred_id":   "DFII5",
        "name":      "5-Year TIPS Constant Maturity Real Rate",
        "sector":    "Real Rates",
        "subsector": "TIPS Real Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DFII7": {
        "fred_id":   "DFII7",
        "name":      "7-Year TIPS Constant Maturity Real Rate",
        "sector":    "Real Rates",
        "subsector": "TIPS Real Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DFII10": {
        "fred_id":   "DFII10",
        "name":      "10-Year TIPS Constant Maturity Real Rate",
        "sector":    "Real Rates",
        "subsector": "TIPS Real Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DFII20": {
        "fred_id":   "DFII20",
        "name":      "20-Year TIPS Constant Maturity Real Rate",
        "sector":    "Real Rates",
        "subsector": "TIPS Real Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "DFII30": {
        "fred_id":   "DFII30",
        "name":      "30-Year TIPS Constant Maturity Real Rate",
        "sector":    "Real Rates",
        "subsector": "TIPS Real Rate",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "T5YIE": {
        "fred_id":   "T5YIE",
        "name":      "5-Year Breakeven Inflation Rate",
        "sector":    "Real Rates",
        "subsector": "Breakeven Inflation",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "T10YIE": {
        "fred_id":   "T10YIE",
        "name":      "10-Year Breakeven Inflation Rate",
        "sector":    "Real Rates",
        "subsector": "Breakeven Inflation",
        "exchange":  "FRED",
        "frequency": "daily",
    },

    # ── Credit spreads (ICE BofA / BAML) ─────────────────────────────────────
    "BAMLH0A0HYM2": {
        "fred_id":   "BAMLH0A0HYM2",
        "name":      "ICE BofA US High Yield Index Option-Adjusted Spread",
        "sector":    "Credit Spreads",
        "subsector": "High Yield OAS",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "BAMLH0A0HYM2EY": {
        "fred_id":   "BAMLH0A0HYM2EY",
        "name":      "ICE BofA US High Yield Index Effective Yield",
        "sector":    "Credit Spreads",
        "subsector": "High Yield Effective Yield",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "BAMLC0A0CM": {
        "fred_id":   "BAMLC0A0CM",
        "name":      "ICE BofA US Corporate Index Option-Adjusted Spread",
        "sector":    "Credit Spreads",
        "subsector": "Investment Grade OAS",
        "exchange":  "FRED",
        "frequency": "daily",
    },
    "BAMLC0A0CMEY": {
        "fred_id":   "BAMLC0A0CMEY",
        "name":      "ICE BofA US Corporate Index Effective Yield",
        "sector":    "Credit Spreads",
        "subsector": "Investment Grade Effective Yield",
        "exchange":  "FRED",
        "frequency": "daily",
    },

    # ── Financial conditions / stress ─────────────────────────────────────────
    "STLFSI4": {
        "fred_id":   "STLFSI4",
        "name":      "St. Louis Fed Financial Stress Index",
        "sector":    "Financial Conditions",
        "subsector": "Financial Stress Index",
        "exchange":  "FRED",
        "frequency": "weekly",
    },
}


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Add a FRED macro series to dim_ticker and backfill fact_price_daily. "
            "Pass --ticker with a key from the built-in registry or any raw FRED series ID."
        )
    )
    p.add_argument("--ticker",      required=True,
                   help="Key from FRED_SERIES registry OR raw FRED series ID")
    p.add_argument("--start-date",  default="2010-01-01", help="YYYY-MM-DD")
    p.add_argument("--end-date",    default="",           help="YYYY-MM-DD (default=today)")
    p.add_argument("--schema",      default=os.getenv("SCHEMA", "analytics"))
    p.add_argument("--ticker-type", default="Macro",
                   help="ticker_type for dim_ticker (default: Macro)")
    p.add_argument("--sleep",       type=float, default=0.0,
                   help="Optional sleep between API calls (seconds)")
    return p.parse_args()


args        = _parse_args()
INPUT_KEY   = args.ticker.strip().upper()
SCHEMA      = args.schema.strip()
TICKER_TYPE = args.ticker_type.strip()
START_DATE  = args.start_date.strip()
END_DATE    = args.end_date.strip() if args.end_date.strip() else _date.today().isoformat()

_reg    = FRED_SERIES.get(INPUT_KEY, {})
FRED_ID = _reg.get("fred_id", INPUT_KEY)
SYMBOL  = INPUT_KEY


# ─────────────────────────────────────────────
# ENV VARS ONLY
# ─────────────────────────────────────────────
FRED_API_KEY = os.environ["FRED_API_KEY"]

PGHOST     = os.environ["PGHOST"]
PGPORT     = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "postgres")
PGUSER     = os.environ["PGUSER"]
PGPASSWORD = os.environ["PGPASSWORD"]
PGSSLMODE  = os.getenv("PGSSLMODE", "require")

ENGINE_URL = (
    f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    f"?sslmode={PGSSLMODE}"
)
engine  = create_engine(ENGINE_URL, pool_pre_ping=True)
pg_conn = psycopg2.connect(
    host=PGHOST, port=PGPORT, dbname=PGDATABASE,
    user=PGUSER, password=PGPASSWORD, sslmode=PGSSLMODE,
)

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "4"))
BACKOFF_BASE    = float(os.getenv("BACKOFF_BASE", "1.6"))


# ─────────────────────────────────────────────
# FRED API helpers
# ─────────────────────────────────────────────
FRED_BASE = "https://api.stlouisfed.org/fred"


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
            print(f"  [WARN] FRED HTTP {r.status_code} for {url}")
            return None
        try:
            return r.json()
        except ValueError:
            return None
    return None


def fetch_fred_series_info(series_id: str) -> Dict[str, Any]:
    js = _get_json(
        f"{FRED_BASE}/series",
        {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"},
    )
    if js and isinstance(js.get("seriess"), list) and js["seriess"]:
        return js["seriess"][0]
    return {}


def fetch_fred_observations(series_id: str, date_from: str, date_to: str) -> pd.DataFrame:
    params: Dict[str, Any] = {
        "series_id":         series_id,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "observation_start": date_from,
        "observation_end":   date_to,
        "units":             "lin",
        "sort_order":        "asc",
    }
    js = _get_json(f"{FRED_BASE}/series/observations", params)
    if not js or not isinstance(js.get("observations"), list):
        return pd.DataFrame()

    rows = []
    for obs in js["observations"]:
        raw_val = obs.get("value", ".")
        if raw_val in (".", "", None):
            continue
        try:
            v = float(raw_val)
        except ValueError:
            continue
        rows.append({"dt": obs["date"], "value": v})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    df = (
        df.sort_values("dt")
        .drop_duplicates(subset=["dt"], keep="last")
        .reset_index(drop=True)
    )
    return df


# ─────────────────────────────────────────────
# 1) RESOLVE METADATA
# ─────────────────────────────────────────────
print(f"[1/4] Resolving metadata for {SYMBOL} (FRED id: {FRED_ID}) ...")

fred_meta = fetch_fred_series_info(FRED_ID)
if args.sleep:
    time.sleep(args.sleep)

reg = FRED_SERIES.get(INPUT_KEY, {})

ticker_name   = reg.get("name")      or fred_meta.get("title")                     or FRED_ID
ticker_sector = reg.get("sector")    or "Macro"
ticker_sub    = reg.get("subsector") or fred_meta.get("seasonal_adjustment_short") or None
ticker_exch   = reg.get("exchange")  or "FRED"
fred_freq     = reg.get("frequency") or fred_meta.get("frequency_short", "").lower() or ""

print(f"  name      : {ticker_name}")
print(f"  sector    : {ticker_sector}")
print(f"  frequency : {fred_freq or 'native'}")


# ─────────────────────────────────────────────
# 2) UPSERT dim_ticker
# ─────────────────────────────────────────────
print(f"[2/4] Upserting dim_ticker for {SYMBOL} ...")

dim_row = {
    "ticker":         SYMBOL,
    "name":           ticker_name,
    "market_cap":     None,
    "beta":           None,
    "employees":      None,
    "industry":       ticker_sub,
    "headquarters":   None,
    "exchange":       ticker_exch,
    "gics_sector":    ticker_sector,
    "gics_subsector": ticker_sub,
    "ticker_type":    TICKER_TYPE,
}

with pg_conn.cursor() as cur:
    cur.execute(
        f"""
        INSERT INTO {SCHEMA}.dim_ticker
          (ticker, name, market_cap, beta, employees, industry,
           headquarters, exchange, gics_sector, gics_subsector, ticker_type)
        VALUES
          (%(ticker)s, %(name)s, %(market_cap)s, %(beta)s, %(employees)s, %(industry)s,
           %(headquarters)s, %(exchange)s, %(gics_sector)s, %(gics_subsector)s, %(ticker_type)s)
        ON CONFLICT (ticker)
        DO UPDATE SET
          name           = COALESCE(EXCLUDED.name,           {SCHEMA}.dim_ticker.name),
          industry       = COALESCE(EXCLUDED.industry,       {SCHEMA}.dim_ticker.industry),
          exchange       = COALESCE(EXCLUDED.exchange,       {SCHEMA}.dim_ticker.exchange),
          gics_sector    = COALESCE(EXCLUDED.gics_sector,    {SCHEMA}.dim_ticker.gics_sector),
          gics_subsector = COALESCE(EXCLUDED.gics_subsector, {SCHEMA}.dim_ticker.gics_subsector),
          ticker_type    = EXCLUDED.ticker_type
        RETURNING ticker_id;
        """,
        dim_row,
    )
    ticker_id = cur.fetchone()[0]
pg_conn.commit()

print(f"  ticker_id = {ticker_id}")


# ─────────────────────────────────────────────
# 3) DOWNLOAD + UPSERT HISTORY
# ─────────────────────────────────────────────
print(f"[3/4] Downloading observations {START_DATE}..{END_DATE} for {FRED_ID} ...")

obs = fetch_fred_observations(FRED_ID, START_DATE, END_DATE)

if obs.empty:
    print(f"[STOP] No observations returned for {FRED_ID} in {START_DATE}..{END_DATE}. Exiting.")
    pg_conn.close()
    engine.dispose()
    raise SystemExit(0)

print(f"  rows fetched: {len(obs):,}  ({obs['dt'].min()} .. {obs['dt'].max()})")

# close = adj_close = FRED value.  open / high / low / volume = NULL.
price_rows = [
    (int(ticker_id), row.dt, None, None, None, row.value, row.value, None)
    for row in obs.itertuples(index=False)
]

print(f"  upserting {len(price_rows):,} rows into fact_price_daily ...")

with pg_conn.cursor() as cur:
    execute_values(
        cur,
        f"""
        INSERT INTO {SCHEMA}.fact_price_daily
          (ticker_id, dt, open, high, low, close, adj_close, volume)
        VALUES %s
        ON CONFLICT (ticker_id, dt)
        DO UPDATE SET
          close     = EXCLUDED.close,
          adj_close = EXCLUDED.adj_close;
        """,
        price_rows,
        page_size=5000,
    )
pg_conn.commit()

print(f"  upserted: {len(price_rows):,} rows")


# ─────────────────────────────────────────────
# 4) REFRESH snapshot_metric_latest
# ─────────────────────────────────────────────
print("[4/4] Refreshing snapshot_metric_latest for this ticker (if applicable) ...")

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
          GROUP BY 1, 2
        ) mx
          ON f.ticker_id = mx.ticker_id
         AND f.metric_id = mx.metric_id
         AND f.dt::date   = mx.dt
        WHERE f.ticker_id = %s
        ON CONFLICT (ticker_id, metric_id)
        DO UPDATE SET
          dt    = EXCLUDED.dt,
          value = EXCLUDED.value
        WHERE ({SCHEMA}.snapshot_metric_latest.dt    IS DISTINCT FROM EXCLUDED.dt)
           OR ({SCHEMA}.snapshot_metric_latest.value IS DISTINCT FROM EXCLUDED.value);
        """,
        (int(ticker_id), int(ticker_id)),
    )
pg_conn.commit()

print(f"[DONE] {SYMBOL} (FRED: {FRED_ID}) added + backfilled ({START_DATE}..{END_DATE}).")
print(f"       {len(price_rows):,} rows in fact_price_daily (close = adj_close = FRED value).")


# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────
try:
    pg_conn.close()
except Exception:
    pass
try:
    engine.dispose()
except Exception:
    pass
