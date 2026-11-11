"""Append-only as-observed daily breadth ledger and dated constituent snapshots.

Historical breadth derived from today's ticker list is *not* automatically a
point-in-time backtest. Archive starts when this workflow is enabled.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_breadth.csv"
CONSTITUENTS = ROOT / "data" / "constituents"
FIELDS = ["date", "universe", "breadth", "passed", "valid", "total", "constituents_sha256"]


def get_payload() -> dict:
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script id="market-data" type="application/json">([\s\S]*?)</script>',
        page
    )
    if not match:
        raise RuntimeError("index.html 缺少 market-data JSON，停止歸檔")
    return json.loads(match.group(1))


def main() -> None:
    data = get_payload()
    latest = data["snapshotDate"]
    expected = os.getenv("EXPECT_TRADE_DATE", "").strip()
    if expected and latest != expected:
        raise RuntimeError(f"資料日 {latest} != 應有交易日 {expected}，禁止歸檔")

    symbols = []
    with (ROOT / "tickers.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            symbols.append(row["Symbol"])
    if len(symbols) != 640 or len(set(symbols)) != 640:
        raise RuntimeError("640 大股票名單不完整，禁止歸檔")

    sha = hashlib.sha256(("\n".join(symbols) + "\n").encode()).hexdigest()
    CONSTITUENTS.mkdir(parents=True, exist_ok=True)
    snapshot = CONSTITUENTS / f"{latest}-{sha[:12]}.csv"
    if not snapshot.exists():
        with snapshot.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Symbol"])
            writer.writerows((symbol,) for symbol in symbols)

    current = {}
    if DAILY.exists():
        with DAILY.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                current[(row["date"], row["universe"])] = row

    i = data["dates"].index(latest)
    added = 0
    for name, series in data["universes"].items():
        value = series["values"][i]
        valid = series["valid"][i]
        passed = series["counts"][i]
        total = int(series["total"])
        if value is None or valid is None or int(valid) < total * .95:
            print(f"跳過 {name}：有效率不足 95%，不新增每日紀錄", flush=True)
            continue
        key = (latest, name)
        if key in current:
            if abs(float(current[key]["breadth"]) - float(value)) > 1e-6:
                print(f"⚠️ {latest} {name} 已有正式紀錄且數值不同：不自動改寫舊值", flush=True)
            continue
        current[key] = {
            "date": latest,
            "universe": name,
            "breadth": f"{value:.8f}",
            "passed": str(int(passed)),
            "valid": str(int(valid)),
            "total": str(total),
            "constituents_sha256": sha if name in ("300", "640") else "official-current-list",
        }
        added += 1

    DAILY.parent.mkdir(parents=True, exist_ok=True)
    temporary = DAILY.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(current[key] for key in sorted(current))
    os.replace(temporary, DAILY)
    print(f"✅ 交易日 {latest}：新增 {added} 筆每日寬度紀錄；股票清單 {snapshot.name}", flush=True)


if __name__ == "__main__":
    main()
