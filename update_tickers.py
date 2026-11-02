
import os
import re
import time
import math
import traceback
from io import StringIO
from pathlib import Path

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
MIN_COVERAGE = 0.90   # 至少取得 90% 股票的有效市值
REQUEST_DELAY = 0.10

SUMMARY_FILE = os.environ.get("GITHUB_STEP_SUMMARY")


class UpdateError(Exception):
    def __init__(self, stage, message, suggestion):
        super().__init__(message)
        self.stage = stage
        self.suggestion = suggestion


def write_summary(lines):
    """將結果寫進 GitHub Actions Summary。"""
    if not SUMMARY_FILE:
        return

    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def safe_text(value):
    """避免錯誤文字破壞 Markdown 表格。"""
    return str(value).replace("|", "/").replace("\n", " ")[:500]


# =========================================================
# 1. 從證交所取得上市、上櫃普通股
# =========================================================

def fetch_taiwan_stock_symbols():
    print("階段 1：取得上市及上櫃普通股清單", flush=True)

    session = requests.Session()

    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
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

        print(f"正在取得{market_name}清單...", flush=True)

        try:
            response = session.get(url, timeout=40)
            response.raise_for_status()
            response.encoding = (
                response.apparent_encoding or "utf-8"
            )

        except requests.RequestException as e:
            raise UpdateError(
                "官方清單下載",
                f"{market_name}網站請求失敗：{str(e)[:300]}",
                "確認證交所網站是否正常，稍後重試。"
            ) from e

        # 核心修正：
        # 使用 StringIO，避免 pandas 把 HTML 誤認成檔案
        try:
            tables = pd.read_html(
                StringIO(response.text),
                header=None
            )
        except Exception as e:
            raise UpdateError(
                "官方 HTML 解析",
                f"{market_name} HTML 解析失敗：{str(e)[:300]}",
                "檢查官方網站表格是否改版，以及 pandas/lxml 版本。"
            ) from e

        selected = None

        # 自動尋找具有 CFICode 的表格與標題列
        for table in tables:
            df = table.copy()
            df.columns = [str(c).strip() for c in df.columns]

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
                    df = df.iloc[row_idx + 1:].copy()
                    selected = df
                    break

            if selected is not None:
                break

        if selected is None:
            raise UpdateError(
                "官方表格欄位辨識",
                f"{market_name}找不到 CFICode 欄位",
                "證交所表格格式可能改變，需檢查欄位名稱。"
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
                "股票代號辨識",
                f"{market_name}找不到股票代號欄位",
                "檢查官方表格第一欄的名稱。"
            )

        # CFICode = ESVUFR：普通股
        df = df[
            df["CFICode"]
            .astype(str)
            .str.strip()
            .eq("ESVUFR")
        ].copy()

        # 只接受四位數股票代號
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
                "普通股清單檢查",
                f"{market_name}沒有取得有效普通股",
                "檢查 CFICode 與股票代號格式。"
            )

        print(
            f"{market_name}：{len(symbols)} 檔",
            flush=True
        )

        return symbols

    twse = get_symbols("2", ".TW", "上市")
    tpex = get_symbols("4", ".TWO", "上櫃")

    all_symbols = list(dict.fromkeys(twse + tpex))

    if len(all_symbols) < TOP_N:
        raise UpdateError(
            "清單數量驗證",
            f"全市場只有 {len(all_symbols)} 檔",
            "股票數量異常，停止更新以保護原資料。"
        )

    print(
        f"官方普通股總數：{len(all_symbols)}",
        flush=True
    )

    return all_symbols, len(twse), len(tpex)


# =========================================================
# 2. 取得 Yahoo Finance 市值
# =========================================================

def get_top_640_by_market_cap(symbols):
    print("階段 2：取得股票市值", flush=True)

    market_caps = []
    failed_symbols = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            mc = info.get("market_cap")

            if mc is None:
                mc = getattr(info, "market_cap", None)

            mc = float(mc) if mc is not None else 0

            if math.isfinite(mc) and mc > 0:
                market_caps.append({
                    "Symbol": symbol,
                    "MarketCap": mc
                })
            else:
                failed_symbols.append(symbol)

        except Exception:
            failed_symbols.append(symbol)

        if i % 100 == 0 or i == total:
            print(
                f"進度 {i}/{total}，"
                f"成功 {len(market_caps)}，"
                f"失敗 {len(failed_symbols)}",
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

    if valid_count < TOP_N or coverage < MIN_COVERAGE:
        raise UpdateError(
            "Yahoo 市值取得率",
            f"有效 {valid_count}/{total} 檔，"
            f"覆蓋率 {coverage:.1%}，"
            f"未達 {MIN_COVERAGE:.0%} 門檻",
            "Yahoo Finance 可能發生限流或缺資料；"
            "請稍後重試，或改採分批及市值快取方式。"
        )

    df = pd.DataFrame(market_caps)

    df = (
        df.drop_duplicates("Symbol")
        .sort_values(
            by=["MarketCap", "Symbol"],
            ascending=[False, True]
        )
        .head(TOP_N)
        .reset_index(drop=True)
    )

    if len(df) != TOP_N:
        raise UpdateError(
            "市值排序驗證",
            f"排序後僅有 {len(df)} 檔股票",
            "檢查是否有重複代號或異常市值。"
        )

    return df, valid_count, failed_symbols, coverage


# =========================================================
# 3. 輸出 CSV
# =========================================================

def save_tickers(df):
    print("階段 3：匯出股票清單", flush=True)

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
        ):
            raise ValueError("CSV 驗證不通過")

        # 驗證成功後才取代正式檔案
        os.replace(temp_file, OUTPUT)

    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()

        raise UpdateError(
            "CSV 寫入",
            str(e),
            "檢查 CSV 欄位、權限及檔案內容。"
        ) from e

    print(f"成功更新：{OUTPUT}", flush=True)


# =========================================================
# 4. 主程式及 GitHub Actions 錯誤摘要
# =========================================================

def main():
    started = time.monotonic()

    try:
        symbols, twse_count, tpex_count = (
            fetch_taiwan_stock_symbols()
        )

        (
            df,
            valid_count,
            failed_symbols,
            coverage
        ) = get_top_640_by_market_cap(symbols)

        save_tickers(df)

        last = df.iloc[-1]
        elapsed = time.monotonic() - started

        print("\n股票名單更新成功", flush=True)
        print(f"市值前 {TOP_N} 大股票")
        print(f"第 {TOP_N} 名：{last['Symbol']}")
        print(f"耗時：{elapsed:.1f} 秒")

        write_summary([
            "## ✅ 股票清單更新成功",
            "",
            "| 項目 | 結果 |",
            "|---|---|",
            f"| 上市普通股 | {twse_count} |",
            f"| 上櫃普通股 | {tpex_count} |",
            f"| 官方總數 | {len(symbols)} |",
            f"| 有效市值 | {valid_count} |",
            f"| 市值取得率 | {coverage:.1%} |",
            f"| 市值排名名單 | {TOP_N} 檔 |",
            f"| 第 {TOP_N} 名 | {last['Symbol']} |",
            f"| 第 {TOP_N} 名市值 | "
            f"{last['MarketCap']/1e8:.2f} 億（Yahoo 回傳幣別） |",
            f"| 抓取失敗數 | {len(failed_symbols)} |",
            f"| 執行時間 | {elapsed:.1f} 秒 |",
            "",
            "注意：市值缺漏可能影響實際排名；"
            "覆蓋率達標不等於完整市場排名。",
            "",
            "### 部分缺漏股票",
            ", ".join(failed_symbols[:30])
            if failed_symbols else "無"
        ])

    except UpdateError as e:
        print(f"\n❌ 更新失敗：{e.stage}", flush=True)
        print(str(e), flush=True)

        write_summary([
            "## ❌ 股票清單更新失敗",
            "",
            "| 項目 | 內容 |",
            "|---|---|",
            f"| 失敗階段 | {safe_text(e.stage)} |",
            f"| 錯誤原因 | {safe_text(e)} |",
            f"| 建議處理 | {safe_text(e.suggestion)} |",
            "| 原始 CSV | 未在本次失敗流程中覆蓋 |"
        ])

        raise SystemExit(1)

    except Exception as e:
        print("\n❌ 未預期錯誤", flush=True)
        traceback.print_exc(limit=5)

        write_summary([
            "## ❌ 未預期錯誤",
            "",
            f"**錯誤類型：** `{type(e).__name__}`",
            "",
            f"**錯誤內容：** {safe_text(e)}",
            "",
            "建議檢查 GitHub Actions Log，"
            "或確認相關套件版本。"
        ])

        raise SystemExit(1)


if __name__ == "__main__":
    main()
