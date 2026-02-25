from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import os
import subprocess
import sys
import io
import threading
import traceback
from datetime import date as _date

app = FastAPI()
bearer = HTTPBearer(auto_error=False)

TRIGGER_SECRET = os.getenv("TRIGGER_SECRET", "")


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def _check_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    if not TRIGGER_SECRET:
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
# ADD TICKER
# ─────────────────────────────────────────────
class TickerRequest(BaseModel):
    ticker: str
    tickerType: str
    startDate: str
    endDate: Optional[str] = ""


@app.post("/add-ticker")
def add_ticker(req: TickerRequest):
    cmd = [
        sys.executable, "add_ticker_backfill.py",
        "--ticker", req.ticker,
        "--ticker-type", req.tickerType,
        "--start-date", req.startDate,
    ]
    if req.endDate:
        cmd += ["--end-date", req.endDate]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy()
    )

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr or "Script failed")

    return {"message": result.stdout or "Ticker added successfully"}


# ─────────────────────────────────────────────
# DAILY RUNNER (FMP) — async with status polling
# ─────────────────────────────────────────────
_daily_last_run: dict = {"status": "idle", "log": "", "date": None}
_daily_lock = threading.Lock()


@app.post("/run-daily")
def run_daily(
    date: Optional[str] = None,
    _auth=Depends(_check_auth),
):
    global _daily_last_run

    target_date = date.strip() if date else _date.today().isoformat()
    try:
        _date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date: {target_date!r}. Use YYYY-MM-DD.")

    with _daily_lock:
        if _daily_last_run.get("status") == "running":
            raise HTTPException(status_code=409, detail="A daily run is already in progress. Check /daily-status.")
        _daily_last_run = {"status": "running", "log": "", "date": target_date}

    thread = threading.Thread(target=_execute_daily, args=(target_date,), daemon=True)
    thread.start()

    return {
        "status": "accepted",
        "target_date": target_date,
        "message": f"Daily runner started for {target_date}. Poll /daily-status for progress.",
    }


@app.get("/daily-status")
def daily_status(_auth=Depends(_check_auth)):
    return _daily_last_run


def _execute_daily(target_date: str):
    global _daily_last_run
    try:
        result = subprocess.run(
            [sys.executable, "dailyrunner.py", "--target-date", target_date]
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=900,
        )
        log = result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else "")
        if result.returncode != 0:
            with _daily_lock:
                _daily_last_run.update({"status": "error", "log": log[-5000:]})
        else:
            with _daily_lock:
                _daily_last_run.update({"status": "success", "log": log[-5000:]})
    except subprocess.TimeoutExpired:
        with _daily_lock:
            _daily_last_run.update({"status": "error", "log": "Timed out after 900s"})
    except Exception:
        with _daily_lock:
            _daily_last_run.update({"status": "error", "log": traceback.format_exc()[-5000:]})


# ─────────────────────────────────────────────
# FRED RUNNER — async with status polling
# ─────────────────────────────────────────────
_fred_last_run: dict = {"status": "idle", "log": "", "date": None}
_fred_lock = threading.Lock()


@app.post("/run-fred")
def run_fred(
    date: Optional[str] = None,
    _auth=Depends(_check_auth),
):
    global _fred_last_run

    target_date = date.strip() if date else _date.today().isoformat()
    try:
        _date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date format: {target_date!r}. Use YYYY-MM-DD.")

    with _fred_lock:
        if _fred_last_run.get("status") == "running":
            raise HTTPException(status_code=409, detail="A FRED run is already in progress. Check /fred-status.")
        _fred_last_run = {"status": "running", "log": "", "date": target_date}

    thread = threading.Thread(target=_execute_fred, args=(target_date,), daemon=True)
    thread.start()

    return {
        "status": "accepted",
        "target_date": target_date,
        "message": f"FRED runner started for {target_date}. Poll /fred-status for progress.",
    }


@app.get("/fred-status")
def fred_status(_auth=Depends(_check_auth)):
    return _fred_last_run


def _execute_fred(target_date: str):
    global _fred_last_run

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

        ENGINE_URL = (
            f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
            f"?sslmode={PGSSLMODE}"
        )
        engine  = create_engine(ENGINE_URL, pool_pre_ping=True)
        pg_conn = psycopg2.connect(
            host=PGHOST, port=PGPORT, dbname=PGDATABASE,
            user=PGUSER, password=PGPASSWORD, sslmode=PGSSLMODE,
        )

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
            _fred_last_run.update({"status": "stopped", "log": buf.getvalue()})
            return

        ticker_to_id = dict(zip(dim_macro["ticker"], dim_macro["ticker_id"]))
        tickers      = dim_macro["ticker"].tolist()
        print(f"  Macro tickers found: {len(tickers)}")

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
            _fred_last_run.update({"status": "no_data", "log": buf.getvalue()})
            return

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
        _fred_last_run.update({"status": "success", "updated": len(results), "skipped": len(no_data)})

    except Exception:
        tb = traceback.format_exc()
        print(f"\n[ERROR]\n{tb}")
        _fred_last_run.update({"status": "error", "error": tb})

    finally:
        sys.stdout = old_stdout
        with _fred_lock:
            _fred_last_run["log"] = buf.getvalue()
        pg_conn_ref = locals().get("pg_conn")
        engine_ref  = locals().get("engine")
        if pg_conn_ref:
            try: pg_conn_ref.close()
            except Exception: pass
        if engine_ref:
            try: engine_ref.dispose()
            except Exception: pass
