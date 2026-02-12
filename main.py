from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import subprocess
import sys

app = FastAPI()

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

@app.get("/health")
def health():
    return {"status": "ok"}
