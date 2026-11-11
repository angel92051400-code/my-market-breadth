"""Repair 2025-08..2025-10 missing daily closes with historical TWSE/TPEx quotations.

No interpolation, no silent price replacement, no cross-date ffill.
The repaired archive is persisted only if ≥95% of current 640 symbols have
valid 60-session histories on every targeted date.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from market_cache import CLOSE_PATH, read_cache, merge_prices, _write_cache
from main import load_top, get_universe, download_prices, calc_breadth

ROOT = Path(__file__).resolve().parent
START = "2025-08-01"
END = "2025-10-29"
MIN_COVERAGE = 0.95
REPORT = ROOT / "data" / "repair_2025_report.json"


def number(value):
    if value is None:
        return np.nan
    s = re.sub(r"<[^>]+>", "", str(value)).strip().replace(",", "")
    if s in ("", "--", "---", "－", "X", "除權", "除息", "nan"):
        return np.nan
    try:
        v = float(s)
        return v if math.isfinite(v) and v > 0 else np.nan
    except ValueError:
        return np.nan


def get_table_rows(payload):
    """Support both legacy data9/fields9 and newer tables[].data/fields."""
    sources = []
    if isinstance(payload, dict):
        sources.append(payload)
        sources += [x for x in payload.get("tables", []) if isinstance(x, dict)]
        if "data9" in payload and "fields9" in payload:
            sources.append({"fields": payload["fields9"], "data": payload["data9"]})
    elif isinstance(payload, list):
        sources = payload

    for item in sources:
        if not isinstance(item, dict):
            continue
        fields = [str(x).strip() for x in item.get("fields", [])]
        rows = item.get("data", [])
        if not fields or not isinstance(rows, list):
            continue
        name_candidates = ("證券代號", "股票代號", "代號", "證券代碼")
        close_candidates = ("收盤價", "收盤", "收市")
        stock = next((i for i, v in enumerate(fields) if v in name_candidates), None)
        closing = next((i for i, v in enumerate(fields) if v in close_candidates), None)
        if stock is not None and closing is not None:
            return rows, stock, closing
    raise ValueError("未找到股票代碼與收盤價同時存在的官方行情表")


def normalize_day(payload, day):
    if isinstance(payload, dict):
        reported = str(payload.get("date", "")).replace("-", "").replace("/", "")
        if re.fullmatch(r"\d{8}", reported) and reported != day.replace("-", ""):
            raise ValueError(f"官方回傳日期 {reported} 與要求日期 {day} 不符")


def parse_market(payload, day, suffix):
    normalize_day(payload, day)
    rows, code_col, close_col = get_table_rows(payload)
    values = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or max(code_col, close_col) >= len(row):
            continue
        code = re.sub(r"<[^>]+>", "", str(row[code_col])).strip()
        if not re.fullmatch(r"[1-9]\d{3}", code):
            continue
        close = number(row[close_col])
        if math.isfinite(close):
            values[code + suffix] = close
    if len(values) < 100:
        raise ValueError(f"官方 {day} {suffix} 股票報價僅 {len(values)} 筆")
    return values


def fetch_json(session, url, params):
    response = session.get(url, params=params, timeout=35)
    response.raise_for_status()
    if "json" not in response.headers.get("content-type", "").lower() and not response.text.lstrip().startswith("{"):
        raise RuntimeError("來源未傳回 JSON，可能遭到流量限制")
    return response.json()


def fetch_day(session, day):
    compact = day.replace("-", "")
    try:
        twse = fetch_json(
            session,
            "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
            {"date": compact, "type": "ALLBUT0999", "response": "json"},
        )
        listed = parse_market(twse, day, ".TW")
    except Exception as exc:
        print(f"⚠️ {day} TWSE：{type(exc).__name__}: {exc}", flush=True)
        listed = {}
    time.sleep(0.7)
    try:
        otc = fetch_json(
            session,
            "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc",
            {"date": day.replace("-", "/"), "response": "json"},
        )
        over_the_counter = parse_market(otc, day, ".TWO")
    except Exception as exc:
        print(f"⚠️ {day} TPEx：{type(exc).__name__}: {exc}", flush=True)
        over_the_counter = {}
    time.sleep(0.7)
    return {**listed, **over_the_counter}, {
        "date": day, "twse_count": len(listed), "tpex_count": len(over_the_counter)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args()

    top = load_top()
    symbols, official, _ = get_universe(top)
    existing = read_cache(CLOSE_PATH)
    if existing.empty:
        print("尚無股價快取：一次性建立兩年基礎歷史資料", flush=True)
        existing = download_prices(symbols, period="2y")

    index_file = ROOT / "index.html"
    page = index_file.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="market-data" type="application/json">([\s\S]*?)</script>', page
    )
    if not match:
        raise RuntimeError("既有 index.html 缺少市場日期資料")
    payload = json.loads(match.group(1))
    dates = pd.DatetimeIndex(payload["dates"])
    targeted = dates[(dates >= args.start) & (dates <= args.end)]
    if len(targeted) < 40:
        raise RuntimeError(f"目標交易日期太少：{len(targeted)}")

    # Two-year baseline sometimes has a whole interval with one valid ticker.
    # Only request official market snapshots on the dates that need fixing.
    existing = existing.reindex(index=existing.index.union(targeted).sort_values())
    count_before = existing.reindex(columns=top).notna().sum(axis=1)
    needs_fix = targeted[count_before.reindex(targeted, fill_value=0) < 608]
    print(f"目標 {len(targeted)} 交易日，需要補資料 {len(needs_fix)} 日", flush=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.twse.com.tw/",
    })
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=2, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504)
    )))
    reports = []
    new = pd.DataFrame(index=targeted)
    for i, timestamp in enumerate(needs_fix, 1):
        day = timestamp.strftime("%Y-%m-%d")
        quotes, report = fetch_day(session, day)
        reports.append(report)
        for symbol, price in quotes.items():
            new.loc[timestamp, symbol] = price
        print(f"{i}/{len(needs_fix)} {day} TWSE {report['twse_count']} TPEx {report['tpex_count']}", flush=True)

    # Official raw prices are only used for missing Yahoo values, never to
    # overwrite existing quotations with a different adjustment convention.
    merged = merge_prices(new, existing)
    # Recompute the actual 60-session moving average coverage; a filled price
    # date alone is insufficient for a valid breadth observation.
    _, _, valid, _ = calc_breadth(merged, top, dates)
    coverage = valid.reindex(targeted) / len(top)
    minimum = float(coverage.min())
    report = {
        "target_start": args.start,
        "target_end": args.end,
        "target_sessions": int(len(targeted)),
        "days_repaired": int(len(needs_fix)),
        "min_60ma_coverage": round(minimum, 5),
        "threshold": MIN_COVERAGE,
        "per_day": reports,
        "still_incomplete": [
            {"date": x.strftime("%Y-%m-%d"), "eligible": int(valid.loc[x])}
            for x in targeted if valid.loc[x] < math.ceil(len(top) * MIN_COVERAGE)
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"檢查結果：最低有效率 {minimum:.1%}；不合格日期 {len(report['still_incomplete'])} 天", flush=True)
    if report["still_incomplete"]:
        # Do not save a partial cache or publish fake historical values.
        raise RuntimeError("官方補價後仍不滿足 95% 的 60MA 有效率；保留舊網站，詳見診斷報告")
    _write_cache(CLOSE_PATH, merged)
    print("✅ 2025 年歷史價格已驗證並儲存；可以重新產生網站", flush=True)


if __name__ == "__main__":
    main()
