"""Taiwan market breadth dashboard.

Produces a self-contained dashboard shell in index.html (Plotly JS via CDN).
Keeps breadth and TAIEX Bollinger Bands, adds dashboard metrics and controls.
Data coverage is disclosed; incomplete Yahoo quotes are not represented as full coverage.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent
TOP_FILE = ROOT / "tickers.csv"
FULL_FILE = ROOT / "all_tickers.csv"
OUTPUT = ROOT / "index.html"
PERIOD = os.getenv("MARKET_PERIOD", "2y")
BATCH_SIZE = int(os.getenv("MARKET_BATCH_SIZE", "80"))
FULL_MARKET = os.getenv("ENABLE_FULL_MARKET", "1") == "1"


def clean_symbols(items):
    return list(dict.fromkeys(
        x.strip().upper() for x in items
        if re.fullmatch(r"\d{4}\.(?:TW|TWO)", str(x).strip().upper())
    ))


def load_top():
    if not TOP_FILE.exists():
        raise FileNotFoundError("找不到 tickers.csv；為避免錯用預設股票，已停止更新")
    df = pd.read_csv(TOP_FILE)
    if "Symbol" not in df.columns:
        raise ValueError("tickers.csv 缺少 Symbol 欄位")
    result = clean_symbols(df["Symbol"].dropna().astype(str))
    if len(result) < 300:
        raise ValueError(f"市值名單只有 {len(result)} 檔，至少需要 300 檔")
    return result


def fetch_official_symbols():
    """Use TWSE ISIN listing; retain complete four-digit common-stock symbols."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "text/html,*/*"})
    result = []
    for mode, suffix in (("2", ".TW"), ("4", ".TWO")):
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
        response = session.get(url, timeout=40)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        tables = pd.read_html(StringIO(response.text), header=None)
        matched = False
        for table in tables:
            for n in range(min(10, len(table))):
                row = [str(v).strip() for v in table.iloc[n]]
                if "CFICode" not in row:
                    continue
                table = table.iloc[n + 1:].copy()
                table.columns = row
                code_col = next((c for c in table.columns if "有價證券代號及名稱" in str(c)), None)
                if not code_col:
                    continue
                ordinary = table[table["CFICode"].astype(str).str.strip() == "ESVUFR"]
                code = ordinary[code_col].astype(str).str.strip().str.extract(r"^(\d{4})(?!\d)")[0].dropna()
                result.extend([s + suffix for s in code.tolist()])
                matched = True
                break
            if matched:
                break
        if not matched:
            raise RuntimeError(f"官方市場 {mode} 表格欄位格式改變")
    result = clean_symbols(result)
    if len(result) < 640:
        raise RuntimeError(f"官方名單異常，只有 {len(result)} 檔")
    return result


def get_universe(top):
    if not FULL_MARKET:
        return top, None, "全市場抓取已停用"
    try:
        all_symbols = fetch_official_symbols()
        return clean_symbols(all_symbols + top), all_symbols, "證交所／櫃買中心普通股"
    except Exception as exc:
        print(f"⚠️ 官方全市場清單無法取得：{type(exc).__name__}: {exc}", flush=True)
        if FULL_FILE.exists():
            try:
                full = clean_symbols(pd.read_csv(FULL_FILE)["Symbol"].dropna().astype(str))
                if len(full) >= 640:
                    return clean_symbols(full + top), full, "備用 all_tickers.csv（可能不是最新名單）"
            except Exception:
                pass
        return top, None, "無可用的全市場股票清單"


def yf_download(items, period=PERIOD):
    try:
        return yf.download(items, period=period, interval="1d", group_by="column",
                           auto_adjust=False, threads=False, progress=False,
                           timeout=30)
    except Exception as exc:
        print(f"⚠️ 報價下載錯誤 {items[:2]}...：{type(exc).__name__}: {exc}", flush=True)
        return pd.DataFrame()


def price_panel(data, column, symbols):
    if data is None or data.empty:
        return pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            result = data[column].copy()
        else:
            result = pd.DataFrame({symbols[0]: data[column]})
    except (KeyError, IndexError):
        return pd.DataFrame()
    if isinstance(result, pd.Series):
        result = result.to_frame(name=symbols[0])
    result.columns = [str(c) for c in result.columns]
    result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
    result = result.loc[:, ~result.columns.duplicated()]
    return result.sort_index()


def download_prices(symbols):
    frames = []
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        d = yf_download(batch)
        p = price_panel(d, "Close", batch)
        if not p.empty:
            frames.append(p)
        print(f"股票歷史報價：{min(i+BATCH_SIZE,len(symbols))}/{len(symbols)}，已得 {sum(len(f.columns) for f in frames)} 欄", flush=True)
        if i + BATCH_SIZE < len(symbols):
            time.sleep(0.6)
    if not frames:
        raise RuntimeError("Yahoo Finance 無法下載任何股票價格")
    result = pd.concat(frames, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    return result.apply(pd.to_numeric, errors="coerce").sort_index()


def extract_ohlc(symbol):
    data = yf_download([symbol])
    if data.empty:
        raise RuntimeError(f"{symbol} 指數報價下載失敗")
    fields = {}
    for k in ("Open", "High", "Low", "Close"):
        x = price_panel(data, k, [symbol])
        if x.empty:
            raise RuntimeError(f"{symbol} 沒有 {k} 資料")
        fields[k] = x.iloc[:, 0]
    return pd.DataFrame(fields).dropna(how="all")


def calc_breadth(close, names, reference_index):
    x = close.reindex(index=reference_index, columns=names)
    ma20 = x.rolling(20, min_periods=20).mean()
    ma60 = x.rolling(60, min_periods=60).mean()
    valid = x.notna() & ma60.notna() & ma20.notna()
    passed = (x > ma20) & (x > ma60) & valid
    denominator = valid.sum(axis=1)
    numerator = passed.sum(axis=1)
    breadth = 100 * numerator.div(denominator.replace(0, np.nan))
    return breadth, numerator, denominator, x


def json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def values(series, index):
    return [None if pd.isna(v) else float(v) for v in series.reindex(index).to_numpy()]


def last_good(series):
    z = series.dropna()
    return float(z.iloc[-1]) if len(z) else None


def run_length(flag):
    n = 0
    for value in flag.iloc[::-1].tolist():
        if not bool(value):
            break
        n += 1
    return n


def main():
    top = load_top()
    all_request, official, source_label = get_universe(top)
    close = download_prices(all_request)
    taiex = extract_ohlc("^TWII")
    us = {}
    for symbol, label in (("^GSPC", "S&P 500"), ("^NDX", "Nasdaq-100")):
        try:
            us[label] = extract_ohlc(symbol)["Close"]
        except Exception as exc:
            print(f"⚠️ 無法取得美股 {label}：{exc}", flush=True)

    idx = taiex.index.intersection(close.index)
    if len(idx) < 80:
        raise RuntimeError(f"台股與成分股交集只有 {len(idx)} 日，資料不足")
    taiex = taiex.reindex(idx)
    # NEVER forward-fill before MA calculation: prevents missing data being treated as actual quotes.
    names = {"300": top[:300], "640": top, "all": official or []}
    breadth = {}
    for key, symbols in names.items():
        if not symbols:
            continue
        b, n, d, prices = calc_breadth(close, symbols, idx)
        # Both sample coverage and latest coverage are disclosed.
        latest_count = int(d.iloc[-1])
        series = b.rolling(20, min_periods=20)
        breadth[key] = {
            "breadth": b, "count": n, "valid": d,
            "mean": series.mean(), "std": series.std(),
            "upper": series.mean() + 2 * series.std(),
            "lower": series.mean() - 2 * series.std(),
            "total": len(symbols), "latest_valid": latest_count,
            "prices": prices,
        }
    if "640" not in breadth or not breadth["640"]["breadth"].notna().any():
        raise RuntimeError("640 檔市場寬度無有效資料")

    taiex_bma = taiex["Close"].rolling(20, min_periods=20).mean()
    taiex_bstd = taiex["Close"].rolling(20, min_periods=20).std()
    ret = taiex["Close"].pct_change(fill_method=None) * 100
    available = breadth["640"]["breadth"].dropna()
    last_date = available.index[-1]
    # Keep all series on common TW trading dates, but don't assume NY market closed on same days.
    dates = [x.strftime("%Y-%m-%d") for x in idx]
    payload = {
        "dates": dates,
        "snapshotDate": last_date.strftime("%Y-%m-%d"),
        "officialSource": source_label,
        "candles": {k.lower(): values(taiex[k], idx) for k in ("Open", "High", "Low", "Close")},
        "taiexReturn": values(ret, idx),
        "taiexBB": {"upper": values(taiex_bma + 2*taiex_bstd, idx),
                    "lower": values(taiex_bma - 2*taiex_bstd, idx)},
        "universes": {}, "us": {},
        "thresholds": {"hot":70, "veryHot":80, "cold":20, "veryCold":10},
        "newHighWindow":20,
        "longWindow":250,
    }
    for key, d in breadth.items():
        payload["universes"][key] = {
            "values": values(d["breadth"], idx),
            "upper": values(d["upper"], idx),
            "lower": values(d["lower"], idx),
            "counts": values(d["count"], idx),
            "valid": values(d["valid"], idx),
            "total": d["total"], "latestValid": d["latest_valid"],
        }
    for label, data in us.items():
        # price index on its own sessions, normalized cumulative return from first valid date
        z = data.dropna()
        z = z.loc[z.index >= idx.min()]
        if len(z) > 1:
            payload["us"][label] = {
                "dates": [x.strftime("%Y-%m-%d") for x in z.index],
                "values": [float(x) for x in ((z / z.iloc[0] - 1)*100)]
            }

    # New close-based highs/lows use previous 20 sessions, not intraday highs/lows.
    all_key = "all" if "all" in breadth else "640"
    px = breadth[all_key]["prices"]
    p20_high = px.shift(1).rolling(20, min_periods=20).max()
    p20_low = px.shift(1).rolling(20, min_periods=20).min()
    highs = ((px > p20_high) & px.notna()).sum(axis=1)
    lows = ((px < p20_low) & px.notna()).sum(axis=1)
    high_diff = highs - lows
    extreme = breadth["640"]["breadth"]
    e_hot = extreme >= 70
    e_cold = extreme <= 20
    # Real streaks on last 250 available trading sessions, not number of prior historic occurrences.
    recent_hot, recent_cold = e_hot.tail(250), e_cold.tail(250)
    payload["extra"] = {
        "highs": values(highs, idx),
        "lows": values(lows, idx),
        "highDiff": values(high_diff, idx),
        "highUniverse": all_key,
        "hotStreak": run_length(recent_hot),
        "coldStreak": run_length(recent_cold),
        "hotCount250": int(recent_hot.sum()),
        "coldCount250": int(recent_cold.sum()),
    }
    # All-market used only when a separate official list exists; flag incomplete quotes explicitly.
    if "all" in breadth:
        r = breadth["all"]["latest_valid"] / max(breadth["all"]["total"], 1)
        payload["allStatus"] = "完整報價覆蓋" if r >= 0.98 else f"部分報價（{r:.1%} 覆蓋）"
    else:
        payload["allStatus"] = "沒有全市場資料"

    html = make_html(payload)
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(html, encoding="utf-8")
    os.replace(temp, OUTPUT)
    print(f"✅ {OUTPUT}：台股資料日 {payload['snapshotDate']}，前640有效 {breadth['640']['latest_valid']}/{len(top)}；全市場 {payload['allStatus']}")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"\n## 📈 市場寬度頁面\n\n- 資料日：{payload['snapshotDate']}\n- 前640有效：{breadth['640']['latest_valid']}/{len(top)}\n- 全市場：{payload['allStatus']}\n- 指標：20/60MA、市場寬度BB、大盤BB、20日收盤新高/低、250日紀錄\n")


def make_html(payload):
    # Escape '</script>' so externally sourced content cannot break inline JSON script.
    data_json = json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>台股市場寬度｜市場監測儀表板</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root { color-scheme:dark; --bg:#14171d; --card:#1e222b; --muted:#98a2b2; --fg:#edf0f5; --border:#333943; --green:#25cf83; --red:#ff626b; --blue:#49a6ff; }
*{box-sizing:border-box} body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans TC',sans-serif;margin:0;background:var(--bg);color:var(--fg)}
main{max-width:1480px;margin:0 auto;padding:22px 18px 38px} .header{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:16px}
.header h1{font-size:22px;margin:0 0 6px}.meta{font-size:13px;color:var(--muted)}.tag{border:1px solid #4a515f;border-radius:6px;padding:4px 8px;color:#ced4de;font-size:12px;margin-right:8px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}.card{background:var(--card);border-radius:12px;padding:16px;min-height:125px}.card .label{font-size:13px;color:#bdc5d0;margin-bottom:7px}.card .value{font-size:29px;letter-spacing:-.8px;font-weight:700}.card .delta{font-size:14px;margin-left:8px}.card .note{font-size:12px;line-height:1.5;color:var(--muted);margin-top:5px}.up{color:var(--red)}.down{color:var(--green)}
.toolbar{display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px;margin:12px 0}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:9px}.controls label{font-size:13px;color:#b5bdca}.controls select,.controls input,.controls button{color:#e9edf3;background:#222730;border:1px solid #454c59;border-radius:7px;font-size:13px;padding:7px 9px;min-height:34px}.controls button{cursor:pointer}.controls button.active{background:#32465d;border-color:#69a8ee}.switches{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px}.switches label{font-size:12px;color:#c0c8d3;cursor:pointer}
.chart-wrap{position:relative;width:100%}#chart{width:100%;height:740px}#crosshair{pointer-events:none;display:none;position:absolute;top:0;left:0;width:0;border-left:1px dashed rgba(222,233,247,.75);z-index:8;box-shadow:0 0 2px #000}.fib-controls select{color:#e9edf3;background:#222730;border:1px solid #454c59;border-radius:6px;padding:4px;font-size:12px}.controls button:disabled{opacity:.45;cursor:not-allowed}.hint{color:#8f9bab;font-size:12px;margin:8px 2px 14px}.panel{border:1px solid #292f39;background:#1b1f27;border-radius:12px;padding:16px;margin-top:14px}.panel h2{font-size:15px;font-weight:600;margin:0 0 12px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric .m-label{font-size:12px;color:#b9c3d0}.metric .m-value{font-size:21px;font-weight:700;margin-top:4px}.foot{font-size:12px;color:#98a2b2;line-height:1.7;margin-top:12px}.warning{color:#f4be6a}.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;color:#b6c0cc;font-size:12px}.dot{display:inline-block;height:9px;width:9px;border-radius:100%;margin-right:5px}
@media(max-width:950px){.cards{grid-template-columns:repeat(2,1fr)}.metrics{grid-template-columns:repeat(2,1fr)}#chart{height:700px}}@media(max-width:560px){main{padding:12px 9px}.cards{gap:8px}.card{padding:12px;min-height:120px}.card .value{font-size:22px}.toolbar{flex-direction:column}#chart{height:630px}.header h1{font-size:18px}.metrics{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main>
<div class="header"><div><h1>台股市場寬度監測</h1><div><span class="tag" id="mood">讀取中</span><span class="meta" id="streak-head"></span></div></div><div class="meta" id="asof"></div></div>
<section class="cards">
<div class="card"><div class="label">市值前 640 大</div><div><span class="value" id="c640">--</span><span class="delta" id="d640"></span></div><div class="note" id="n640"></div></div>
<div class="card"><div class="label">全市場 <span id="all-qual"></span></div><div><span class="value" id="call">--</span><span class="delta" id="delta-all"></span></div><div class="note" id="nall"></div></div>
<div class="card"><div class="label">20 日收盤新高－新低</div><div><span class="value" id="cnew">--</span><span class="delta" id="dnew"></span></div><div class="note" id="nnew"></div></div>
<div class="card"><div class="label">大盤高低柱－盤中</div><div><span class="value" id="cprice">--</span><span class="delta" id="dprice"></span></div><div class="note" id="nprice"></div></div>
</section>
<div class="toolbar">
 <div class="controls"><label for="start">顯示區間</label><input type="date" id="start"/><span>—</span><input type="date" id="end"/><button type="button" id="reset">還原</button><button type="button" id="d1m">1 個月</button><button type="button" id="d3m">3 個月</button><button type="button" id="d6m">6 個月</button><button type="button" id="d1y">1 年</button><button type="button" id="dall">全部</button></div>
 <div class="controls"><label for="us-select">美股對照</label><select id="us-select"><option value="none">不顯示</option><option value="S&P 500">S&P 500（報酬%）</option><option value="Nasdaq-100">Nasdaq-100（報酬%）</option></select><label for="primary">台股寬度</label><select id="primary"><option value="640">前 640 大</option><option value="300">前 300 大</option><option value="all">全市場</option></select></div>
</div>
<div class="switches"><label><input type="checkbox" id="show300"/> 前 300 大</label><label><input type="checkbox" id="show640" checked/> 前 640 大</label><label><input type="checkbox" id="showall"/> 全市場</label><label><input type="checkbox" id="taiexbb"/> 大盤布林通道</label><label><input type="checkbox" id="breadthbb"/> 市場寬度布林通道</label><label><input type="checkbox" id="heat" checked/> 過熱 70/80%</label><label><input type="checkbox" id="cold" checked/> 過冷 10/20%</label><label><input type="checkbox" id="newbars" checked/> 新高－新低柱狀</label><label><input type="checkbox" id="fib"/> 費波那契回撤線</label><label class="fib-controls" for="fibdir">波段方向 <select id="fibdir"><option value="auto">自動（依高低點先後）</option><option value="up">上漲波段</option><option value="down">下跌波段</option></select></label></div>
<div class="chart-wrap" id="chart-wrap"><div id="chart"></div><div id="crosshair" aria-hidden="true"></div></div><div class="hint" id="chart-hint">拖曳圖表可縮放、移動；滑鼠參考線會穿過主圖與兩個獨立副圖；日期切換時三圖同步。費波那契以所選日期區間的大盤最高／最低點自動計算，僅是可能價位，不是買賣保證。K 線與市場寬度共用日期軸，左右 Y 軸分別顯示股價點數及市場寬度百分比。美股線為指數累積報酬（非美股市場寬度），使用獨立 Y 軸。</div>
<div class="panel"><h2>門檻與歷史脈絡 <span class="meta" id="coverage"></span></h2><div class="metrics"><div class="metric"><div class="m-label">距過熱門檻（640 大 ≥70%）</div><div class="m-value" id="hotgap">--</div></div><div class="metric"><div class="m-label">距過冷門檻（640 大 ≤20%）</div><div class="m-value" id="coldgap">--</div></div><div class="metric"><div class="m-label">近 250 交易日過熱次數</div><div class="m-value" id="hotcount">--</div></div><div class="metric"><div class="m-label">近 250 交易日過冷次數</div><div class="m-value" id="coldcount">--</div></div></div>
<div class="foot" id="details"></div></div>
</main><script id="market-data" type="application/json">__DATA__</script>
<script>
(() => {
const D=JSON.parse(document.getElementById('market-data').textContent);
const $=id=>document.getElementById(id), n=D.dates.length, last=n-1;
const val=(arr,i=last)=>arr&&arr[i]!=null?arr[i]:null;
const fmt=(x,d=1)=>x==null?'—':Number(x).toLocaleString('zh-TW',{minimumFractionDigits:d,maximumFractionDigits:d});
const sign=(x,d=1)=>x==null?'':(x>0?'+':'')+fmt(x,d);
const delta=(a,b)=>a==null||b==null?null:a-b;
const pct=k=>D.universes[k]||null;
function card(valueId,deltaId,noteId,value,diff,note,unit='%'){
 $(valueId).textContent=value==null?'—':fmt(value,1)+unit;
 const el=$(deltaId);el.textContent=sign(diff,1);el.className='delta '+(diff>0?'up':diff<0?'down':'');$(noteId).textContent=note;
}
const latest=pct('640'); const v640=val(latest.values), v640Prev=val(latest.values,last-1);
const a=pct('all'), av=a?val(a.values):null;
card('c640','d640','n640',v640,delta(v640,v640Prev),`有效 ${fmt(val(latest.valid),0)}/${latest.total} 檔；站上雙均線 ${fmt(val(latest.counts),0)} 檔`);
card('call','delta-all','nall',av,a?delta(av,val(a.values,last-1)):null,a?`有效 ${fmt(val(a.valid),0)}/${a.total} 檔；${D.allStatus}`:'官方全市場報價暫不可用');
$('all-qual').textContent=a?'· '+D.allStatus:'';
const highs=val(D.extra.highs),lows=val(D.extra.lows),diff=val(D.extra.highDiff);
$('cnew').textContent=diff==null?'—':sign(diff,0); $('dnew').textContent='';$('nnew').textContent=`創高 ${fmt(highs,0)} / 創低 ${fmt(lows,0)}；以${D.extra.highUniverse==='all'?'全市場':'前640大'}有效收盤價計算`;
const hi=val(D.candles.high),lo=val(D.candles.low),cl=val(D.candles.close);
$('cprice').textContent=fmt(cl,0);$('dprice').textContent=sign(val(D.taiexReturn),2)+'%';$('nprice').textContent=`當日最高 ${fmt(hi,0)}／最低 ${fmt(lo,0)}；非逐筆即時數據`;
$('asof').textContent='資料日 '+D.snapshotDate;const isHot=v640>=70,isCold=v640<=20;
$('mood').textContent=v640>=80?'極度過熱':isHot?'過熱':v640<=10?'極度過冷':isCold?'過冷':'中性';$('streak-head').textContent=`連續 ${isHot?D.extra.hotStreak:isCold?D.extra.coldStreak:0} 個交易日（同一狀態）`;
$('hotgap').textContent=fmt(v640-70,1)+' 個百分點';$('coldgap').textContent=fmt(v640-20,1)+' 個百分點';$('hotcount').textContent=D.extra.hotCount250+' 日';$('coldcount').textContent=D.extra.coldCount250+' 日';
$('coverage').textContent=`｜全市場來源：${D.officialSource}`;
$('details').textContent='寬度＝收盤價同時高於 20 日與 60 日簡單移動平均線的有效股票比率；歷史布林通道為寬度 20 日平均 ±2 倍標準差，大盤布林通道為加權指數收盤價 20 日平均 ±2 倍標準差。新高／新低為當日收盤價突破先前 20 個有效交易日的收盤高／低，不是盤中高低或 52 週新高。紅漲、綠跌。歷史名單採用目前名單回推，並非各歷史日當時的市值前 640 大，存在成分股／存活者偏差。';
let currentStart=D.dates[Math.max(0,last-99)],currentEnd=D.dates[last];
$('start').min=D.dates[0];$('start').max=D.dates[last];$('end').min=D.dates[0];$('end').max=D.dates[last];
$('start').value=currentStart;$('end').value=currentEnd;
const C={bg:'#14171d',grid:'#343a43',fg:'#a9b4c5',green:'#25cf83',red:'#ff626b',blue:'#49a6ff'};
const chart=$('chart'),crosshair=$('crosshair'),wrap=$('chart-wrap');
let drawing=false,handlersBound=false;
const clampDate=d=>d<D.dates[0]?D.dates[0]:d>D.dates[last]?D.dates[last]:d;
function uiRange(){ $('start').value=currentStart; $('end').value=currentEnd; for(const [id] of [['d1m'],['d3m'],['d6m'],['d1y'],['dall']])$(id).classList.remove('active'); }
function setRange(start,end,button=''){
  if(start>end){$('chart-hint').textContent='開始日期不能晚於結束日期，請重新選擇。';uiRange();return;}
  currentStart=clampDate(start);currentEnd=clampDate(end);
  uiRange();if(button)$(button).classList.add('active');build();
}
function trailingDays(days){
  const date=new Date(D.dates[last]+'T12:00:00Z');date.setUTCDate(date.getUTCDate()-days);
  return clampDate(date.toISOString().slice(0,10));
}
function trace(x,y,name,color,axis='y2',extra={}){return {type:'scatter',mode:'lines',x,y,name,connectgaps:false,line:{color,width:2},yaxis:axis,...extra};}
function fibonacci(){
  const pairs=[];
  for(let i=0;i<n;i++){
    const date=D.dates[i],hi=D.candles.high[i],lo=D.candles.low[i];
    if(date>=currentStart&&date<=currentEnd&&Number.isFinite(hi)&&Number.isFinite(lo))pairs.push({i,hi,lo});
  }
  if(pairs.length<2)return null;
  const highest=pairs.reduce((a,b)=>a.hi>=b.hi?a:b);
  const lowest=pairs.reduce((a,b)=>a.lo<=b.lo?a:b);
  const span=highest.hi-lowest.lo;
  if(!(span>0))return null;
  const requested=$('fibdir').value;
  const direction=requested==='auto'?(lowest.i<=highest.i?'up':'down'):requested;
  // 上漲波段：從高點向下回撤；下跌波段：從低點向上反彈
  const lvls=[0,.236,.382,.5,.618,.786,1].map(level=>({
    level, price:direction==='up'?highest.hi-span*level:lowest.lo+span*level
  }));
  return {lvls,direction,highest,lowest};
}
function hideCrosshair(){crosshair.style.display='none';}
function drawCrosshair(evt){
  if(!evt||!evt.points||!evt.points.length||!chart._fullLayout)return;
  const lay=chart._fullLayout;
  const ax=lay.xaxis,top=lay.yaxis,bottom=lay.yaxis5;
  if(!ax||!top||!bottom)return;
  const date=evt.points[0].x;
  let x;
  try{x=ax._offset+ax.d2p(date);}catch(_){x=NaN;}
  if(!Number.isFinite(x)&&evt.event&&Number.isFinite(evt.event.clientX)){
    x=evt.event.clientX-chart.getBoundingClientRect().left;
  }
  if(!Number.isFinite(x)||x<ax._offset-1||x>ax._offset+ax._length+1){hideCrosshair();return;}
  const y0=Math.min(top._offset,bottom._offset);
  const y1=Math.max(top._offset+top._length,bottom._offset+bottom._length);
  crosshair.style.left=`${x}px`;
  crosshair.style.top=`${y0}px`;
  crosshair.style.height=`${Math.max(0,y1-y0)}px`;
  crosshair.style.display='block';
}
function syncZoom(evt){
  if(!evt||drawing)return;
  let a=evt['xaxis.range[0]'],b=evt['xaxis.range[1]'];
  if((a==null||b==null)&&Array.isArray(evt['xaxis.range']))[a,b]=evt['xaxis.range'];
  if(a==null||b==null){a=evt['xaxis2.range[0]'];b=evt['xaxis2.range[1]'];}
  if(a==null||b==null){a=evt['xaxis3.range[0]'];b=evt['xaxis3.range[1]'];}
  if((a==null||b==null)&&Array.isArray(evt['xaxis2.range']))[a,b]=evt['xaxis2.range'];
  if((a==null||b==null)&&Array.isArray(evt['xaxis3.range']))[a,b]=evt['xaxis3.range'];
  if(a!=null&&b!=null){
    const date=d=>String(d).slice(0,10);
    const aa=clampDate(date(a)),bb=clampDate(date(b));
    if(aa<=bb){currentStart=aa;currentEnd=bb;uiRange();}
  } else if(evt['xaxis.autorange']===true||evt['xaxis2.autorange']===true||evt['xaxis3.autorange']===true){
    currentStart=D.dates[0];currentEnd=D.dates[last];uiRange();
  }
}
async function build(){
 if(!window.Plotly){$('chart-hint').textContent='圖表套件 Plotly 載入失敗，請確認網路可連接 CDN。';return;}
 const ts=[];
 const candles={type:'candlestick',name:'台股加權指數',x:D.dates,open:D.candles.open,high:D.candles.high,low:D.candles.low,close:D.candles.close,increasing:{line:{color:C.red},fillcolor:C.red},decreasing:{line:{color:C.green},fillcolor:C.green},yaxis:'y',xaxis:'x',showlegend:true};ts.push(candles);
 const colors={'300':'#738598','640':'#f4f6fc','all':'#a68aff'};
 for(const [k,name] of [['300','前 300 大'],['640','前 640 大'],['all','全市場']]){
  if(pct(k)&&$('show'+k).checked)ts.push(trace(D.dates,pct(k).values,name,colors[k]));
 }
 const p=pct($('primary').value);
 if(p&&$('breadthbb').checked){
  ts.push(trace(D.dates,p.upper,'寬度布林上軌','#f2ad53','y2',{line:{color:'#f2ad53',width:1,dash:'dash'}}));
  ts.push(trace(D.dates,p.lower,'寬度布林下軌','#f2ad53','y2',{line:{color:'#f2ad53',width:1,dash:'dash'},fill:'tonexty',fillcolor:'rgba(239,165,68,.12)'}));
 }
 if($('taiexbb').checked){
  ts.push(trace(D.dates,D.taiexBB.upper,'大盤布林上軌','#9db4c9','y',{line:{color:'#9db4c9',width:1,dash:'dot'}}));
  ts.push(trace(D.dates,D.taiexBB.lower,'大盤布林下軌','#9db4c9','y',{line:{color:'#9db4c9',width:1,dash:'dot'},fill:'tonexty',fillcolor:'rgba(145,167,189,.09)'}));
 }
 const us=$('us-select').value;
 if(us!=='none'&&D.us[us])ts.push(trace(D.us[us].dates,D.us[us].values,us+' 累積報酬 (%)',us==='S&P 500'?'#f64da8':'#8b68e2','y4'));
 ts.push({type:'bar',name:'大盤日漲跌幅 (%)',x:D.dates,y:D.taiexReturn,xaxis:'x2',yaxis:'y3',marker:{color:D.taiexReturn.map(v=>v==null?'#555':v>=0?C.red:C.green)},showlegend:true});
 if($('newbars').checked)ts.push({type:'bar',name:'20日收盤創高－創低',x:D.dates,y:D.extra.highDiff,xaxis:'x3',yaxis:'y5',opacity:.80,marker:{color:D.extra.highDiff.map(v=>v>=0?'#f3c56a':'#87aaf6')},showlegend:true});
 const shapes=[],annotations=[];
 if($('heat').checked)for(const [level,color,dash] of [[70,'#fb923c','dash'],[80,'#ff626b','solid']])shapes.push({type:'line',xref:'paper',yref:'y2',x0:0,x1:1,y0:level,y1:level,line:{color,width:1.3,dash}});
 if($('cold').checked)for(const [level,color,dash] of [[20,'#4ade80','dash'],[10,'#15803d','solid']])shapes.push({type:'line',xref:'paper',yref:'y2',x0:0,x1:1,y0:level,y1:level,line:{color,width:1.3,dash}});
 if($('fib').checked){
  const fib=fibonacci();
  if(fib){
   for(const lvl of fib.lvls){
    const important=Math.abs(lvl.level-.618)<.001;
    const color=important?'#eec16c':'#7da7d1';
    shapes.push({type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:lvl.price,y1:lvl.price,line:{color,width:important?1.8:1,dash:lvl.level===0||lvl.level===1?'solid':'dash'}});
    annotations.push({xref:'paper',yref:'y',x:1,y:lvl.price,xanchor:'right',yanchor:'bottom',text:`${(lvl.level*100).toFixed(1)}% ${lvl.price.toFixed(0)}`,showarrow:false,font:{size:10,color},bgcolor:'rgba(20,23,29,.76)',borderpad:1});
   }
  }
 }
 const layout={
 paper_bgcolor:C.bg,plot_bgcolor:C.bg,font:{color:C.fg,size:11},
 height:window.innerWidth<600?630:740,margin:{l:56,r:80,t:36,b:85},
 hovermode:'x unified',hoversubplots:'axis',spikedistance:-1,dragmode:'zoom',
 showlegend:true,legend:{orientation:'h',x:0,y:-.16,font:{size:11}},barmode:'relative',
 xaxis:{domain:[0,1],anchor:'y',type:'date',range:[currentStart,currentEnd],showticklabels:false,showgrid:false,showspikes:true,spikemode:'across',spikesnap:'cursor',spikedash:'dot',rangebreaks:[{bounds:['sat','mon']}],rangeslider:{visible:false}},
 xaxis2:{domain:[0,1],anchor:'y3',type:'date',matches:'x',range:[currentStart,currentEnd],showticklabels:false,gridcolor:C.grid,showspikes:true,spikemode:'across',spikesnap:'cursor',spikedash:'dot',rangebreaks:[{bounds:['sat','mon']}],rangeslider:{visible:false}},
 xaxis3:{domain:[0,1],anchor:'y5',type:'date',matches:'x',range:[currentStart,currentEnd],gridcolor:C.grid,showspikes:true,spikemode:'across',spikesnap:'cursor',spikedash:'dot',rangebreaks:[{bounds:['sat','mon']}],rangeslider:{visible:false}},
 yaxis:{domain:[.45,1],side:'right',gridcolor:C.grid,title:{text:'台股點數',font:{size:11}}},
 yaxis2:{overlaying:'y',side:'left',range:[0,100],title:{text:'市場寬度 (%)',font:{size:11}},showgrid:false},
 yaxis3:{domain:[.25,.39],gridcolor:C.grid,zeroline:true,title:{text:'日漲跌 (%)',font:{size:10}}},
 yaxis4:{overlaying:'y',side:'right',position:.92,showgrid:false,title:{text:'美股累積報酬 (%)',font:{size:10}}},
 yaxis5:{domain:[0,.18],gridcolor:C.grid,zeroline:true,title:{text:'20日新高－新低 (檔)',font:{size:10}}},
 shapes,annotations,hoverlabel:{bgcolor:'#232934',font:{color:'#fafafa'}}
 };
 drawing=true;hideCrosshair();
 try{await Plotly.react(chart,ts,layout,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']});}
 catch(err){console.error('Plotly render error',err);$('chart-hint').textContent='圖表繪製失敗：'+String(err);}
 finally{drawing=false;}
 if(!handlersBound&&typeof chart.on==='function'){chart.on('plotly_hover',drawCrosshair);chart.on('plotly_unhover',hideCrosshair);chart.on('plotly_relayout',syncZoom);handlersBound=true;}
}
$('start').addEventListener('change',()=>setRange($('start').value||D.dates[0],$('end').value||D.dates[last]));
$('end').addEventListener('change',()=>setRange($('start').value||D.dates[0],$('end').value||D.dates[last]));
$('reset').onclick=()=>setRange(D.dates[Math.max(0,last-99)],D.dates[last]);
for(const [btn,days] of [['d1m',30],['d3m',90],['d6m',180],['d1y',365]])$(btn).onclick=()=>setRange(trailingDays(days),D.dates[last],btn);
$('dall').onclick=()=>setRange(D.dates[0],D.dates[last],'dall');
for(const id of ['show300','show640','showall','taiexbb','breadthbb','heat','cold','newbars','us-select','primary','fib','fibdir'])$(id).addEventListener('change',build);
if(!pct('all')){$('showall').disabled=true;$('primary').querySelector('[value="all"]').disabled=true;}
for(const k of ['S&P 500','Nasdaq-100'])if(!D.us[k]){const o=$('us-select').querySelector(`[value="${k}"]`);if(o)o.disabled=true;}
// Never attach repeated listeners inside build(): preserves handlers across Plotly.react.
chart.addEventListener('mouseleave',hideCrosshair);
window.addEventListener('resize',hideCrosshair);
build();
})();
</script></body></html>'''
    return template.replace("__DATA__", data_json)


if __name__ == "__main__":
    main()
