
import os
import re
import time
import math
import traceback
from io import StringIO
from pathlib import Path
from collections import Counter

import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================================================
# 0. 基本設定
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "tickers.csv"

TOP_N = 640
MIN_COVERAGE = 0.90

# 每檔查詢之間的延遲
REQUEST_DELAY = 0.20

# 連續失敗達到此數量就提前停止
MAX_CONSECUTIVE_FAILURES = 15

# Log 最多顯示幾筆個別錯誤
MAX_PRINT_ERRORS = 30

SUMMARY_FILE = os.environ.get("GITHUB_STEP_SUMMARY")


class UpdateError(Exception):
    def __init__(self, stage, message, suggestion):
        super().__init__(message)
        self.stage = stage
        self.suggestion = suggestion


def write_summary(lines):
    if not SUMMARY_FILE:
        return

    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def safe_text(value):
    return (
        str(value)
        .replace("|", "/")
        .replace("\n", " ")
        [:500]
    )


# =========================================================
# 1. 取得上市、上櫃普通股清單
# =========================================================

def fetch_taiwan_stock_symbols():
    print("階段 1：取得上市及上櫃普通股", flush=True)

    session = requests.Session()

    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504]
    )

    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry)
    )

    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml"
    })

    def get_symbols(mode, suffix, market_name):
        url = (
            "https://isin.twse.com.tw/isin/"
            f"C_public.jsp?strMode={mode}"
        )

        print(
            f"正在取得{market_name}清單...",
            flush=True
        )

        try:
            response = session.get(url, timeout=40)
            response.raise_for_status()
            response.encoding = (
                response.apparent_encoding or "utf-8"
            )

        except requests.RequestException as e:
            raise UpdateError(
                "官方清單下載",
                f"{market_name}下載失敗：{e}",
                "確認官方網站是否可以正常連線。"
            ) from e

        # 使用 StringIO，修正先前 HTML 解析錯誤
        try:
            tables = pd.read_html(
                StringIO(response.text),
                header=None
            )

        except Exception as e:
            raise UpdateError(
                "官方 HTML 解析",
                f"{market_name} HTML 解析失敗：{e}",
                "檢查證交所表格格式是否改變。"
            ) from e

        selected = None

        for table in tables:
            df = table.copy()
            df.columns = [
                str(c).strip() for c in df.columns
            ]

            if "CFICode" in df.columns:
                selected = df
                break

            for row_idx in range(min(len(df), 10)):
                values = [
                    str(v).strip()
                    for v in df.iloc[row_idx].tolist()
                ]

                if "CFICode" in values:
                    df.columns = values
                    selected = df.iloc[row_idx + 1:].copy()
                    break

            if selected is not None:
                break

        if selected is None:
            raise UpdateError(
                "官方表格欄位辨識",
                f"{market_name}找不到 CFICode 欄位",
                "官方表格可能改版，請檢查欄位。"
            )

        df = selected.copy()

        name_col = next(
            (
                c for c in df.columns
                if "有價證券代號及名稱" in str(c)
            ),
            None
        )

        if name_col is None:
            raise UpdateError(
                "股票代號欄位",
                f"{market_name}找不到股票代號欄位",
                "檢查官方表格欄位名稱。"
            )

        # 只保留普通股
        df = df[
            df["CFICode"]
            .astype(str)
            .str.strip()
            .eq("ESVUFR")
        ].copy()

        codes = (
            df[name_col]
            .astype(str)
            .str.strip()
            .str.extract(r"^(\d{4})(?!\d)")[0]
            .dropna()
        )

        symbols = (
            codes + suffix
        ).drop_duplicates().tolist()

        if not symbols:
            raise UpdateError(
                "普通股名單",
                f"{market_name}沒有有效普通股",
                "檢查 CFICode 和股票代號格式。"
            )

        print(
            f"{market_name}：{len(symbols)} 檔",
            flush=True
        )

        return symbols

    twse = get_symbols("2", ".TW", "上市")
    tpex = get_symbols("4", ".TWO", "上櫃")

    symbols = list(dict.fromkeys(twse + tpex))

    if len(symbols) < TOP_N:
        raise UpdateError(
            "清單數量驗證",
            f"只取得 {len(symbols)} 檔股票",
            "股票清單異常，停止更新。"
        )

    print(
        f"官方普通股總數：{len(symbols)}",
        flush=True
    )

    return symbols, len(twse), len(tpex)


# =========================================================
# 2. Yahoo Finance 市值查詢
# =========================================================

def get_top_640_by_market_cap(symbols):
    print("階段 2：取得股票市值", flush=True)

    market_caps = []
    failures = []
    error_counts = Counter()

    total = len(symbols)
    consecutive_failures = 0

    for i, symbol in enumerate(symbols, 1):
        error_type = None
        error_message = ""

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            market_cap = info.get("market_cap")

            if market_cap is None:
                market_cap = getattr(
                    info, "market_cap", None
                )

            market_cap = (
                float(market_cap)
                if market_cap is not None
                else 0
            )

            if not math.isfinite(market_cap) or market_cap <= 0:
                error_type = "InvalidMarketCap"
                error_message = "市值為空值、零或無效數值"

            else:
                market_caps.append({
                    "Symbol": symbol,
                    "MarketCap": market_cap
                })
                consecutive_failures = 0

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)[:250]

        if error_type is not None:
            failures.append({
                "Symbol": symbol,
                "ErrorType": error_type,
                "Message": error_message
            })

            error_counts[error_type] += 1
            consecutive_failures += 1

            # 只列出部分錯誤，避免 Log 過長
            if len(failures) <= MAX_PRINT_ERRORS:
                print(
                    f"[FAILED] {i}/{total} "
                    f"{symbol} | {error_type} | "
                    f"{error_message}",
                    flush=True
                )

            # 如果偵測到明確限流，立即停止
            rate_limit_detected = (
                "RateLimit" in error_type
                or "429" in error_message
                or "Too Many Requests" in error_message
            )

            if rate_limit_detected:
                raise UpdateError(
                    "Yahoo Finance 頻率限制",
                    f"第 {i} 檔 {symbol} 發生限制："
                    f"{error_type} - {error_message}",
                    "停止本次市值更新，"
                    "保留舊名單並繼續每日市場寬度更新。"
                )

            # 如果連續大量失敗，提前停止
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                recent_failures = failures[
                    -MAX_CONSECUTIVE_FAILURES:
                ]

                examples = ", ".join(
                    item["Symbol"]
                    for item in recent_failures[:5]
                )

                raise UpdateError(
                    "Yahoo 連續查詢失敗",
                    f"在第 {i} 檔附近連續失敗 "
                    f"{consecutive_failures} 次。"
                    f"例如：{examples}。"
                    f"錯誤統計：{dict(error_counts)}",
                    "可能為限流、暫時性連線問題"
                    "或某區段股票資料異常；"
                    "請檢查個別錯誤訊息。"
                )

        if i % 100 == 0 or i == total:
            print(
                f"進度 {i}/{total}，"
                f"成功 {len(market_caps)}，"
                f"失敗 {len(failures)}",
                flush=True
            )

        time.sleep(REQUEST_DELAY)

    valid_count = len(market_caps)
    coverage = valid_count / total

    print(
        f"有效市值：{valid_count}/{total}"
        f"（{coverage:.1%}）",
        flush=True
    )

    if (
        valid_count < TOP_N
        or coverage < MIN_COVERAGE
    ):
        raise UpdateError(
            "Yahoo 市值取得率",
            f"有效 {valid_count}/{total} 檔，"
            f"覆蓋率 {coverage:.1%}，"
            f"未達 {MIN_COVERAGE:.0%} 門檻。"
            f"錯誤分類：{dict(error_counts)}",
            "不降低覆蓋率門檻，"
            "保留原始 CSV，"
            "使用舊清單更新市場寬度。"
        )

    df = pd.DataFrame(market_caps)

    df = (
        df.drop_duplicates("Symbol")
        .sort_values(
            ["MarketCap", "Symbol"],
            ascending=[False, True]
        )
        .head(TOP_N)
        .reset_index(drop=True)
    )

    return df, valid_count, failures, coverage


# =========================================================
# 3. 安全儲存 CSV
# =========================================================

def save_tickers(df):
    print("階段 3：儲存 tickers.csv", flush=True)

    temp_file = OUTPUT.with_suffix(".tmp")

    try:
        df[["Symbol"]].to_csv(
            temp_file,
            index=False,
            encoding="utf-8"
        )

        check = pd.read_csv(temp_file)

        if (
            len(check) != TOP_N
            or check["Symbol"].isna().any()
            or check["Symbol"].duplicated().any()
            or not check["Symbol"].astype(str)
                .str.fullmatch(r"\d{4}\.(TW|TWO)")
                .all()
        ):
            raise ValueError("CSV 資料驗證失敗")

        os.replace(temp_file, OUTPUT)

    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()

        raise UpdateError(
            "CSV 儲存失敗",
            str(e),
            "請確認欄位、檔案格式及寫入權限。"
        ) from e

    print("tickers.csv 更新成功", flush=True)


# =========================================================
# 4. 主程式和 GitHub Summary
# =========================================================

def main():
    start_time = time.monotonic()

    try:
        symbols, twse_count, tpex_count = (
            fetch_taiwan_stock_symbols()
        )

        (
            df,
            valid_count,
            failures,
            coverage
        ) = get_top_640_by_market_cap(symbols)

        save_tickers(df)

        last = df.iloc[-1]
        elapsed = time.monotonic() - start_time

        print("✅ 市值前 640 大更新成功")

        write_summary([
            "## ✅ 股票市值排名更新成功",
            "",
            "| 項目 | 結果 |",
            "|---|---|",
            f"| 上市普通股 | {twse_count} |",
            f"| 上櫃普通股 | {tpex_count} |",
            f"| 官方總數 | {len(symbols)} |",
            f"| 有效市值 | {valid_count} |",
            f"| 覆蓋率 | {coverage:.1%} |",
            f"| 更新股票數 | {TOP_N} |",
            f"| 第 640 名 | {last['Symbol']} |",
            f"| 耗時 | {elapsed:.1f} 秒 |",
            "",
            "### 失敗股票（最多 30 檔）",
            "",
            *[
                f"- `{x['Symbol']}`："
                f"{safe_text(x['ErrorType'])}"
                for x in failures[:30]
            ],
            "",
            "注意：覆蓋率達標仍無法保證"
            "市值前 640 大完全沒有缺漏。"
        ])

    except UpdateError as e:
        print(
            f"\n❌ {e.stage}\n{e}",
            flush=True
        )

        write_summary([
            "## ❌ 股票清單更新失敗",
            "",
            "| 項目 | 內容 |",
            "|---|---|",
            f"| 失敗階段 | {safe_text(e.stage)} |",
            f"| 原因 | {safe_text(e)} |",
            f"| 建議 | {safe_text(e.suggestion)} |",
            "| tickers.csv | 未覆蓋原有檔案 |",
            "",
            "後續工作流程將驗證舊名單，"
            "合格時繼續更新市場寬度圖表。"
        ])

        raise SystemExit(1)

    except Exception as e:
        traceback.print_exc(limit=5)

        write_summary([
            "## ❌ 未預期的程式錯誤",
            "",
            f"錯誤類型：`{type(e).__name__}`",
            "",
            f"內容：{safe_text(e)}"
        ])

        raise SystemExit(1)


if __name__ == "__main__":
    main()
