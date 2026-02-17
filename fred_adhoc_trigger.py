# jobs/fred_adhoc_trigger.py
# ==========================
# AD-HOC HTTP TRIGGER FOR FRED MACRO DAILY RUNNER
#
# Exposes a FastAPI endpoint so you can trigger the FRED runner for any
# target date from a phone / browser — no terminal needed.
#
# Deploy alongside your existing Railway service (or as its own service).
#
# ENDPOINTS
#   GET  /health                      — liveness check
#   POST /run-fred?date=2026-02-13    — run for a specific date
#   POST /run-fred                    — run for today
#
# SECURITY
#   Set TRIGGER_SECRET env var. Pass it as Bearer token:
#     Authorization: Bearer <your-secret>
#   Without this header the endpoint returns 401.
#
# USAGE (from phone browser or curl)
#   curl -X POST "https://your-railway-app.up.railway.app/run-fred?date=2026-02-13" \
#        -H "Authorization: Bearer YOUR_SECRET"
#
# INSTALL
#   pip install fastapi uvicorn
#   uvicorn jobs.fred_adhoc_trigger:app --host 0.0.0.0 --port 8000
# ==========================

import os
import io
import sys
import threading
import traceback
from datetime import date as _date
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ── import the runner as a callable module ──────────────────────────────────
# The runner logic is extracted into run_fred_for_date() below so this file
# is self-contained. If you prefer, you can refactor fred_daily_runner.py to
# expose a run(target_date, schema) function and import it instead.

app    = FastAPI(title="FRED Adhoc Trigger", version="1.0.0")
bearer = HTTPBearer(auto_error=False)

TRIGGER_SECRET = os.getenv("TRIGGER_SECRET", "")


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def _check_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    if not TRIGGER_SECRET:
        # No secret configured — open access (not recommended for production)
        return True
    if credentials is None or credentials.credentials != TRIGGER_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")
    return True


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# TRIGGER ENDPOINT
# ─────────────────────────────────────────────
# Stores the last run result so you can poll /status
_last_run: dict = {"status": "idle", "log": "", "date": None}
_lock = threading.Lock()


@app.post("/run-fred")
def run_fred(
    date: Optional[str] = Query(
        default=None,
        description="Target date in YYYY-MM-DD format. Defaults to today.",
        example="2026-02-13",
    ),
    _auth=Depends(_check_auth),
):
    """
    Trigger the FRED macro daily runner for a given date.

    - Pass ?date=YYYY-MM-DD for a historical or missed date.
    - Leave blank to run for today.
    - Returns immediately with a 202 and streams output into /status.
    """
    global _last_run

    # Validate date
    target_date = date.strip() if date else _date.today().isoformat()
    try:
        _date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date format: {target_date!r}. Use YYYY-MM-DD.")

    with _lock:
        if _last_run.get("status") == "running":
            raise HTTPException(status_code=409, detail="A run is already in progress. Check /status.")
        _last_run = {"status": "running", "log": "", "date": target_date}

    # Run in background thread so we return 202 immediately
    thread = threading.Thread(target=_execute_runner, args=(target_date,), daemon=True)
    thread.start()

    return {
        "status":      "accepted",
        "target_date": target_date,
        "message":     f"FRED runner started for {target_date}. Poll /status for progress.",
    }


@app.get("/status")
def status(_auth=Depends(_check_auth)):
    """Returns the result of the last /run-fred call."""
    return _last_run


# ─────────────────────────────────────────────
# RUNNER LOGIC
# Mirrors fred_daily_runner.py — env vars are read at call time.
# ─────────────────────────────────────────────
def _execute_runner(target_date: str):
    """Runs the FRED macro pipeline for target_date. Captures stdout to _last_run."""
    global _last_run

    # Capture all print() output
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    try:
        import time
        import requests
        import pandas as pd
        from sqlalchemy import create_engine, text
        import psycopg2
        from psycopg2.extras import execute_values

        T_DATE = _date.fromisoformat(target_date)

        # ── env vars ──────────────────────────────────────────────────────
        FRED_API_KEY = os.environ["FRED_API_KEY"]
        PGHOST       = os.environ["PGHOST"]
        PGPORT       = int(os.getenv("PGPORT", "5432"))
        PGDATABASE   = os.getenv("PGDATABASE", "postgres")
        PGUSER       = os.environ["PGUSER"]
        PGPASSWORD   = os.environ["PGPASSWORD"]
        PGSSLMODE    = os.getenv("PGSSLMODE", "require")
        SCHEMA       = os.getenv("SCHEMA", "analytics")
        REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "4"))
        BACKOFF_BASE    = float(os.getenv("BACKOFF_BASE", "1.6"))
        FRED_BASE       = "https://api.stlouisfed.org/fred"

        print(f"FRED adhoc runner — target date: {target_date}")

        # ── connect ───────────────────────────────────────────────────────
        ENGINE_URL = (
            f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
            f"?sslmode={PGSSLMODE}"
        )
        engine  = create_engine(ENGINE_URL, pool_pre_ping=True)
        pg_conn = psycopg2.connect(
            host=PGHOST, port=PGPORT, dbname=PGDATABASE,
            user=PGUSER, password=PGPASSWORD, sslmode=PGSSLMODE,
        )

        # ── [1/4] load macro universe ─────────────────────────────────────
        print(f"\n[1/4] Loading Macro tickers from {SCHEMA}.dim_ticker ...")
        dim_macro = pd.read_sql(
            text(f"""
                SELECT ticker_id, ticker
                FROM {SCHEMA}.dim_ticker
                WHERE ticker_type = 'Macro' AND ticker IS NOT NULL
                ORDER BY ticker
            """),
            engine,
        )
        if dim_macro.empty:
            print("[STOP] No Macro tickers found in dim_ticker.")
            _last_run.update({"status": "stopped", "log": buf.getvalue()})
            return

        ticker_to_id = dict(zip(dim_macro["ticker"], dim_macro["ticker_id"]))
        tickers      = dim_macro["ticker"].tolist()
        print(f"  Macro tickers found: {len(tickers)}")

        # ── FRED helpers ──────────────────────────────────────────────────
        def _get_json(url, params):
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
                        delay = float(r.headers.get("Retry-After") or BACKOFF_BASE ** attempt)
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

        def fetch_observation_for_date(series_id, tgt_date):
            params = {
                "series_id":         series_id,
                "api_key":           FRED_API_KEY,
                "file_type":         "json",
                "observation_start": tgt_date,
                "observation_end":   tgt_date,
                "units":             "lin",
                "sort_order":        "asc",
            }
            js = _get_json(f"{FRED_BASE}/series/observations", params)
            if not js or not isinstance(js.get("observations"), list):
                return None
            for obs in js["observations"]:
                raw_val = obs.get("value", ".")
                if raw_val in (".", "", None):
                    continue
                if obs.get("date") != tgt_date:
                    continue
                try:
                    return float(raw_val)
                except ValueError:
                    continue
            return None

        # ── [2/4] fetch observations ──────────────────────────────────────
        print(f"\n[2/4] Fetching FRED observations for {target_date} ...")
        results, no_data = [], []
        for i, ticker in enumerate(tickers, start=1):
            value = fetch_observation_for_date(ticker, target_date)
            if value is None:
                no_data.append(ticker)
                print(f"  [{i:>3}/{len(tickers)}] {ticker:<22} — no data")
            else:
                results.append({
                    "ticker": ticker, "ticker_id": ticker_to_id[ticker],
                    "dt": T_DATE, "value": value,
                })
                print(f"  [{i:>3}/{len(tickers)}] {ticker:<22} = {value}")

        print(f"\n  Got data : {len(results)} tickers")
        print(f"  No data  : {len(no_data)} tickers")

        if not results:
            print(f"\n[STOP] No FRED observations for {target_date} (weekend/holiday?).")
            pg_conn.close(); engine.dispose()
            _last_run.update({"status": "no_data", "log": buf.getvalue()})
            return

        # ── [3/4] upsert fact_price_daily ─────────────────────────────────
        print(f"\n[3/4] Upserting {len(results)} rows ...")
        price_rows = [
            (r["ticker_id"], r["dt"], None, None, None, r["value"], r["value"], None)
            for r in results
        ]
        with pg_conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {SCHEMA}.fact_price_daily
                  (ticker_id, dt, open, high, low, close, adj_close, volume)
                VALUES %s
                ON CONFLICT (ticker_id, dt)
                DO UPDATE SET close = EXCLUDED.close, adj_close = EXCLUDED.adj_close;
                """,
                price_rows,
                page_size=5000,
            )
        pg_conn.commit()
        print(f"  Upserted: {len(price_rows)} rows")

        # ── [4/4] refresh snapshot_metric_latest ─────────────────────────
        print(f"\n[4/4] Refreshing snapshot_metric_latest ...")
        updated_ids = [r["ticker_id"] for r in results]
        with pg_conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {SCHEMA}.snapshot_metric_latest (ticker_id, metric_id, dt, value)
                SELECT f.ticker_id, f.metric_id, f.dt::date, f.value
                FROM {SCHEMA}.fact_metric_daily f
                JOIN (
                  SELECT ticker_id, metric_id, MAX(dt)::date AS dt
                  FROM {SCHEMA}.fact_metric_daily
                  WHERE ticker_id = ANY(%s)
                  GROUP BY 1, 2
                ) mx
                  ON f.ticker_id = mx.ticker_id
                 AND f.metric_id = mx.metric_id
                 AND f.dt::date   = mx.dt
                WHERE f.ticker_id = ANY(%s)
                ON CONFLICT (ticker_id, metric_id)
                DO UPDATE SET dt = EXCLUDED.dt, value = EXCLUDED.value
                WHERE (snapshot_metric_latest.dt    IS DISTINCT FROM EXCLUDED.dt)
                   OR (snapshot_metric_latest.value IS DISTINCT FROM EXCLUDED.value);
                """,
                (updated_ids, updated_ids),
            )
        pg_conn.commit()

        print(f"\n[DONE] {len(results)} tickers updated, {len(no_data)} skipped.")
        _last_run.update({"status": "success", "updated": len(results), "skipped": len(no_data)})

    except Exception:
        tb = traceback.format_exc()
        print(f"\n[ERROR]\n{tb}")
        _last_run.update({"status": "error", "error": tb})

    finally:
        sys.stdout = old_stdout
        with _lock:
            _last_run["log"] = buf.getvalue()
        pg_conn_ref = locals().get("pg_conn")
        engine_ref  = locals().get("engine")
        if pg_conn_ref:
            try: pg_conn_ref.close()
            except Exception: pass
        if engine_ref:
            try: engine_ref.dispose()
            except Exception: pass
