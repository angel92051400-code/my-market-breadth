
import os
import re
import time
import pandas as pd
import requests
import yfinance as yf
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TOP_N = 640
MIN_VALID_CAPS = 640
OUTPUT = Path(__file__).resolve().parent / "tickers.csv"

def fetch_taiwan_stock_symbols():
    print("1. 取得上市、上櫃普通股名單")

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": "Mozilla/5.0"
    })

    def get_symbols(mode, suffix):
        url = (
            "https://isin.twse.com.tw/isin/"
            f"C_public.jsp?strMode={mode}"
        )
        response = session.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        tables = pd.read_html(
            response.text,
            header=0
        )
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]

        # 相容網站表格第一列才是欄名的情況
        if "CFICode" not in df.columns:
            df.columns = df.iloc[0].astype(str).str.strip()
            df = df.iloc[1:].copy()

        if "CFICode" not in df.columns:
            raise ValueError("找不到 CFICode 欄位")

        name_col = next(
            (c for c in df.columns
             if "有價證券代號及名稱" in c),
            None
        )
        if name_col is None:
            raise ValueError("找不到股票代號欄位")

        df = df[
            df["CFICode"].astype(str).str.strip()
            == "ESVUFR"
        ].copy()

        codes = (
            df[name_col]
            .astype(str)
            .str.strip()
            .str.extract(r"^(\d{4})")[0]
            .dropna()
        )

        return (codes + suffix).tolist()

    symbols = list(dict.fromkeys(
        get_symbols("2", ".TW") +
        get_symbols("4", ".TWO")
    ))

    if len(symbols) < 640:
        raise RuntimeError(
            f"官方清單只有 {len(symbols)} 檔，停止更新"
        )

    print(f"取得 {len(symbols)} 檔普通股")
    return symbols


def get_top_640_by_market_cap(symbols):
    print("2. 開始抓取市值")
    market_caps = []
    failures = []

    for i, sym in enumerate(symbols, 1):
        try:
            ticker = yf.Ticker(sym)
            mc = ticker.fast_info.get("market_cap")

            if mc is not None and mc > 0:
                market_caps.append({
                    "Symbol": sym,
                    "MarketCap": float(mc)
                })
            else:
                failures.append(sym)

        except Exception as e:
            failures.append(sym)

        if i % 100 == 0:
            print(
                f"處理 {i}/{len(symbols)}，"
                f"成功 {len(market_caps)}"
            )

        time.sleep(0.1)

    df = pd.DataFrame(market_caps)

    if len(df) < MIN_VALID_CAPS:
        raise RuntimeError(
            f"只有 {len(df)} 檔取得有效市值，"
            "不覆蓋原本 tickers.csv"
        )

    if failures:
        print(f"警告：{len(failures)} 檔缺少有效市值")
        print("部分失敗代號：", failures[:20])

    df = (
        df.sort_values("MarketCap", ascending=False)
          .drop_duplicates("Symbol")
          .head(TOP_N)
          .reset_index(drop=True)
    )

    return df


if __name__ == "__main__":
    symbols = fetch_taiwan_stock_symbols()
    result = get_top_640_by_market_cap(symbols)

    # 完成全部驗證後才覆蓋正式 CSV
    temp_file = OUTPUT.with_suffix(".tmp")
    result[["Symbol"]].to_csv(
        temp_file, index=False
    )
    os.replace(temp_file, OUTPUT)

    last = result.iloc[-1]
    print(f"成功更新 {OUTPUT}")
    print(f"總檔數：{len(result)}")
    print(
        f"第 640 名：{last['Symbol']}，"
        f"市值約 {last['MarketCap']/1e8:.2f} 億"
    )
