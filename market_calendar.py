"""TWSE trading-calendar gate. Uses the official current-year OpenAPI.

If the calendar cannot be verified, return UNKNOWN (never invent a holiday).
The dashboard's quote-date check provides a second safety gate.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

URL = "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule"


def roc_to_iso(value: str) -> str:
    digits = "".join(c for c in str(value) if c.isdigit())
    if len(digits) != 7:
        raise ValueError(f"Invalid ROC date: {value!r}")
    return f"{int(digits[:3]) + 1911:04d}-{digits[3:5]}-{digits[5:7]}"


def is_closed_record(record: dict) -> bool:
    name = str(record.get("Name", "")).strip()
    # These informational rows denote real trading days.
    if "最後交易日" in name or "開始交易日" in name or "開始交易" in name:
        return False
    # Holidays and settlement-only sessions are not trading days.
    return bool(name)


def trading_status(day: date, records: list[dict] | None) -> tuple[str, str]:
    if records is None:
        return ("closed", "weekend") if day.weekday() >= 5 else ("unknown", "官方日曆無法確認")
    relevant = [r for r in records if roc_to_iso(r.get("Date", "")) == day.isoformat()]
    if relevant:
        if any(is_closed_record(r) for r in relevant):
            return "closed", "；".join(str(r.get("Name", "")) for r in relevant)
        return "open", "官方列示交易日"
    if day.weekday() >= 5:
        return "closed", "weekend"
    return "open", "官方日曆未列為休市"


def fetch_calendar() -> list[dict]:
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=25) as response:
        records = json.load(response)
    if not isinstance(records, list) or not records:
        raise RuntimeError("TWSE calendar response is not a list")
    if not all(isinstance(x, dict) and "Date" in x and "Name" in x for x in records):
        raise RuntimeError("TWSE calendar fields changed")
    return records


def run() -> str:
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    day = now.date()
    try:
        records = fetch_calendar()
        # The OpenAPI sometimes exposes only the latest published year.
        years = {int(str(x["Date"])[:3]) + 1911 for x in records}
        if day.year not in years:
            raise RuntimeError(f"Calendar year {day.year} missing (available: {years})")
        status, reason = trading_status(day, records)
    except Exception as exc:
        print(f"TWSE calendar unavailable: {type(exc).__name__}: {exc}", flush=True)
        status, reason = trading_status(day, None)
    print(f"Taipei {day}: {status} - {reason}", flush=True)

    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"status={status}\n")
            f.write(f"date={day.isoformat()}\n")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"## 台股交易日檢查\n\n- 台灣日期：{day}\n- 狀態：{status}\n- 說明：{reason}\n\n")
    return status


if __name__ == "__main__":
    run()
