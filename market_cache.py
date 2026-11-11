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


def load_stock_closes(symbols: list[str], downloader, *, full_refresh=False) -> pd.DataFrame:
    existing = read_cache(CLOSE_PATH)
    if existing.empty or full_refresh:
        print("歷史股價：首次建檔／定期完整校驗，下載 2y", flush=True)
        fresh = downloader(symbols, period="2y")
    else:
        print("歷史股價：增量更新近 1mo", flush=True)
        fresh = downloader(symbols, period="1mo")

    merged = merge_prices(existing, fresh)
    new_symbols = [s for s in symbols if s not in merged.columns]
    if new_symbols:
        print(f"新上市或新入選成分股 {len(new_symbols)} 檔：補抓 2y", flush=True)
        merged = merge_prices(merged, downloader(new_symbols, period="2y"))

    merged = merged.reindex(columns=symbols)
    if len(merged) < MIN_HISTORY_ROWS:
        raise RuntimeError(f"股價快取只有 {len(merged)} 個交易日，禁止發布")
    if merged.notna().sum(axis=1).iloc[-1] < max(300, int(len(symbols) * .90)):
        raise RuntimeError("最新交易日股價覆蓋率不足 90%，保留舊網站")
    return merged


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
