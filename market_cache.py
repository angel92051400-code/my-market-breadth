"""Persistent incremental daily close/TAIEX cache for the market breadth dashboard.

Cache uses pandas gzip CSV (no pickle), committed by GitHub Actions after a
successful update. Existing data are never replaced with missing Yahoo values.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "data"
CLOSE_PATH = CACHE_DIR / "stock_closes.csv.gz"
TAIEX_PATH = CACHE_DIR / "taiex_ohlc.csv.gz"
MIN_HISTORY_ROWS = 80


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
    x = x.loc[~x.index.duplicated(keep="last")]
    x = x.loc[:, ~x.columns.duplicated(keep="last")]
    return x.apply(pd.to_numeric, errors="coerce").sort_index()


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return _normalize(pd.read_csv(path, index_col=0, parse_dates=True, compression="gzip"))
    except Exception as exc:
        raise RuntimeError(f"無法讀取歷史快取 {path}: {exc}") from exc


def merge_prices(old: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    old, fresh = _normalize(old), _normalize(fresh)
    if old.empty:
        return fresh
    if fresh.empty:
        return old
    # Prefer a new non-null price; never let null overwrite a known quote.
    return fresh.combine_first(old).sort_index()


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    try:
        _normalize(frame).to_csv(temporary, compression="gzip", float_format="%.10g")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _quote_status(frame: pd.DataFrame, symbols: list[str], target: pd.Timestamp) -> tuple[int, list[str]]:
    """A column does not imply a valid quotation on the actual target date."""
    row = (frame.loc[target] if target in frame.index else
           pd.Series(dtype=float))
    valid = [symbol for symbol in symbols
             if symbol in row.index and pd.notna(row[symbol]) and row[symbol] > 0]
    valid_set = set(valid)
    return len(valid), [symbol for symbol in symbols if symbol not in valid_set]


def _save_quote_diagnostics(target: pd.Timestamp, symbols: list[str],
                            missing: list[str], top: list[str],
                            attempts: list[dict], latest_seen: str | None) -> None:
    """Persist diagnostic evidence even when the dashboard is not published."""
    import json
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    top_set = set(top)
    missing_top = [symbol for symbol in missing if symbol in top_set]
    data = {
        "target_date": target.strftime("%Y-%m-%d"),
        "latest_downloaded_date": latest_seen,
        "total_symbols": len(symbols),
        "valid": len(symbols)-len(missing),
        "coverage_pct": round(100*(len(symbols)-len(missing))/len(symbols), 3),
        "top640_valid": len(top)-len(missing_top),
        "top640_total": len(top),
        "missing_count": len(missing),
        "missing_top_count": len(missing_top),
        "missing_tickers": missing,
        "attempts": attempts,
    }
    path = CACHE_DIR/"quote_diagnostics.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"行情品質診斷 {path}: {data['valid']}/{len(symbols)} "
          f"({data['coverage_pct']:.2f}%), "
          f"前640: {data['top640_valid']}/{len(top)}, "
          f"缺價 {len(missing)} 檔", flush=True)
    if missing:
        print("缺價代號範例：" + ", ".join(missing[:30]), flush=True)


def load_stock_closes(symbols: list[str], downloader, *, full_refresh=False,
                      expected_date: str = "", priority_symbols: list[str] | None = None,
                      max_retry_symbols: int = 320) -> pd.DataFrame:
    """Download, verify the intended date and retry only missing target-day closes.

    Never forward-fill, change the date, silently relax coverage requirements or
    persist unvalidated data. The caller writes caches only after full validation.
    """
    import math
    existing = read_cache(CLOSE_PATH)
    priority = [s for s in (priority_symbols or []) if s in symbols]
    missing_history = [symbol for symbol in symbols if symbol not in existing.columns]
    if existing.empty or full_refresh:
        print("歷史股價：首次建檔／定期完整校驗，下載 2y", flush=True)
        fresh = downloader(symbols, period="2y")
    else:
        print("歷史股價：增量更新近 1mo", flush=True)
        fresh = downloader(symbols, period="1mo")

    seen_dates = fresh.index if fresh is not None and not fresh.empty else existing.index
    latest_seen = str(pd.Timestamp(seen_dates.max()).date()) if len(seen_dates) else None
    target = (pd.Timestamp(expected_date).normalize()
              if expected_date else
              pd.Timestamp(seen_dates.max()).normalize() if len(seen_dates)
              else None)
    if target is None:
        raise RuntimeError("Yahoo沒有提供任何交易日股價，保留舊網站")
    if target in existing.index and expected_date and target > pd.Timestamp.today().normalize():
        raise ValueError("預期交易日不能在未來")
    print(f"股價預定交易日：{target.date()}，下載最新日期：{latest_seen}", flush=True)

    merged = merge_prices(existing, fresh)
    if missing_history and not existing.empty and not full_refresh:
        print(f"新上市或新入選成分股 {len(missing_history)} 檔：補抓 2y", flush=True)
        merged = merge_prices(merged, downloader(missing_history, period="2y"))

    # Never treat an early quote from the *next* date as the current-day
    # coverage test. Preserve earlier historical observations.
    merged = merged.loc[merged.index <= target].reindex(columns=symbols)
    if len(merged) < MIN_HISTORY_ROWS:
        raise RuntimeError(f"股價快取只有 {len(merged)} 個交易日，禁止發布")

    needed = max(300, math.ceil(len(symbols)*.90))
    top_needed = math.ceil(len(priority)*.95) if priority else 0
    attempts=[]
    for retry_round in range(3):
        count, missing = _quote_status(merged, symbols, target)
        top_count = len(priority) - len(set(missing).intersection(priority))
        print(f"交易日 {target.date()}：全市場 {count}/{len(symbols)} "
              f"({count/len(symbols):.2%})，前640 {top_count}/{len(priority)}",
              flush=True)
        if count >= needed and top_count >= top_needed:
            _save_quote_diagnostics(target, symbols, missing, priority, attempts, latest_seen)
            return merged

        if retry_round == 2:
            break
        # Prioritise the 640 constituents; retry only names with no valid close
        # on the intended date, even if Yahoo reported columns for those names.
        missing_set = set(missing)
        ordered = [s for s in priority if s in missing_set]
        ordered.extend(s for s in missing if s not in set(ordered))
        candidates = ordered[:max_retry_symbols]
        if not candidates:
            break
        # If the target date is absent entirely, retry one small batch first.
        # Repeating thousands of tickers cannot manufacture a closed session.
        if count == 0:
            candidates = candidates[:80]
        print(f"缺價補抓第 {retry_round+1} 輪：{len(candidates)} 檔，"
              f"原缺漏 {len(missing)} 檔，僅重試預定交易日缺價者", flush=True)
        attempts.append({"round":retry_round+1, "requested":len(candidates),
                         "before_valid":count})
        retry = downloader(candidates, period="1mo")
        if retry is not None and not retry.empty:
            merged = merge_prices(merged, retry.loc[retry.index <= target])
            merged = merged.reindex(columns=symbols)

    count, missing = _quote_status(merged, symbols, target)
    top_count = len(priority)-len(set(missing).intersection(priority))
    _save_quote_diagnostics(target, symbols, missing, priority, attempts, latest_seen)
    reasons = []
    if count < needed:
        reasons.append(f"全市場 {count}/{len(symbols)} < {needed} (90%)")
    if priority and top_count < top_needed:
        reasons.append(f"前640 {top_count}/{len(priority)} < {top_needed} (95%)")
    raise RuntimeError(
        f"交易日 {target.date()} 報價不足：{'; '.join(reasons)}。"
        "已針對缺價股票重試；保留舊網站，詳見 data/quote_diagnostics.json"
    )


def load_taiex(fetcher, *, full_refresh=False) -> pd.DataFrame:
    old = read_cache(TAIEX_PATH)
    fresh = fetcher("2y" if old.empty or full_refresh else "1mo")
    out = merge_prices(old, fresh)
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(out.columns):
        raise RuntimeError("加權指數快取缺少 OHLC 欄位")
    if len(out) < MIN_HISTORY_ROWS:
        raise RuntimeError("加權指數歷史日數不足")
    return out[list(("Open", "High", "Low", "Close"))]


def persist(closes: pd.DataFrame, taiex: pd.DataFrame) -> None:
    _write_cache(CLOSE_PATH, closes)
    _write_cache(TAIEX_PATH, taiex)
    print(f"✅ 股價快取已儲存：{CLOSE_PATH} / {TAIEX_PATH}", flush=True)


def validate_and_mask_breadth(breadth, counts, total: int, *, coverage=.95):
    """Do not turn 1/1 into 100%; retain per-date counts for diagnostic display."""
    minimum = max(1, int(__import__("math").ceil(total * coverage)))
    masked = breadth.where(counts >= minimum)
    removed = int((breadth.notna() & masked.isna()).sum())
    if removed:
        print(f"⚠️ 歷史市場寬度：{removed} 天少於 {minimum}/{total} 有效股票，標記為缺值", flush=True)
    return masked, removed
