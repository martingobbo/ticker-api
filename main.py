from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

class TickerRequest(BaseModel):
    ticker: str
    tickerType: str
    startDate: str

@app.post("/add-ticker")
def add_ticker(req: TickerRequest):
    result = subprocess.run(
        ["python3", "add_ticker_backfill.py",
         "--ticker", req.ticker,
         "--ticker-type", req.tickerType,
         "--start-date", req.startDate],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return {"error": result.stderr}, 500
    return {"message": result.stdout or "Ticker added successfully"}
