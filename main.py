import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

print("啟動量化引擎：開始下載資料與計算市場寬度...")

# ==========================================
# 1. 取得台股加權指數歷史資料 (改為 2 年，解決均線助跑期問題)
# ==========================================
print("📥 步驟 1: 正在下載大盤加權指數 (^TWII)...")
taiex_data = yf.download("^TWII", period="2y", progress=False)
taiex_close = taiex_data['Close']
# 計算大盤每日漲跌幅 (%)
taiex_pct = taiex_close.pct_change() * 100

# ==========================================
# 2. 自動讀取 650 檔股票名單 (tickers.csv)
# ==========================================
csv_file = 'tickers.csv'
if os.path.exists(csv_file):
    print(f"📁 步驟 2: 找到 {csv_file}，正在讀取股票代號...")
    ticker_df = pd.read_csv(csv_file)
    
    if 'Symbol' in ticker_df.columns:
        tickers = ticker_df['Symbol'].astype(str).tolist()
    else:
        tickers = ticker_df.iloc[:, 0].astype(str).tolist()
else:
    print(f"⚠️ 找不到 {csv_file}！使用預設的台灣前 10 大權值股名單...")
    tickers = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", 
               "2882.TW", "2891.TW", "3231.TW", "2382.TW", "2356.TW"]

tickers = [t.strip() for t in tickers if t.strip()]
print(f"🎯 準備抓取 {len(tickers)} 檔股票資料！")

# ==========================================
# 3. 批次下載成分股歷史報價 (同步改為 2 年)
# ==========================================
print("📥 步驟 3: 正在從 Yahoo Finance 批次下載個股資料...")
stock_data = yf.download(tickers, period="2y", threads=True, progress=False)

if isinstance(stock_data.columns, pd.MultiIndex):
    close_prices = stock_data['Close']
else:
    close_prices = pd.DataFrame(stock_data['Close'])

# ==========================================
# 4. 核心量化邏輯與修復
# ==========================================
print("🧮 步驟 4: 進行均線與市場寬度極速運算...")
ma20 = close_prices.rolling(window=20).mean()
ma60 = close_prices.rolling(window=60).mean()

above_both = (close_prices > ma20) & (close_prices > ma60)
breadth_series = above_both.mean(axis=1) * 100

# ⚠️ 關鍵修復：把前 60 天因為算不出季線而變成 0% 的「助跑期」強制轉為空值 (NaN)
breadth_series[ma60.isna().all(axis=1)] = pd.NA

# ==========================================
# 5. 資料合併與清理
# ==========================================
print("🧹 步驟 5: 對齊時間軸與整理漲跌色彩...")
df = pd.DataFrame({
    'taiex': taiex_close.squeeze(),
    'taiex_pct': taiex_pct.squeeze(),
    'breadth': breadth_series.squeeze()
})

# 剔除所有的空值日 (包含前 60 天的助跑期)
df = df.dropna()
df = df.reset_index()
date_col = df.columns[0]

# 根據台灣股市習慣，設定柱狀圖顏色 (漲=紅色，跌=綠色)
df['bar_color'] = np.where(df['taiex_pct'] >= 0, 'red', 'green')

# ==========================================
# 6. Plotly 雙層互動圖表生成
# ==========================================
print("📊 步驟 6: 正在生成子圖疊加 HTML 互動圖表...")

# 建立上下兩層的版面 (75% 主圖，25% 副圖)，共用 X 軸
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    row_heights=[0.75, 0.25],
    vertical_spacing=0.08,
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
)

# ----------------- 上層 (Row 1) -----------------
# 藍線：加權指數
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['taiex'], name="加權指數", mode='lines', line=dict(color='blue', width=2)),
    row=1, col=1, secondary_y=False
)
# 橘線：市場寬度
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['breadth'], name="市場寬度 (%)", mode='lines', line=dict(color='orange', width=2)),
    row=1, col=1, secondary_y=True
)

# 繪製紅綠警戒線 (僅在上層副 Y 軸)
for y_val, color in zip([20, 30, 50, 60], ['green', 'green', 'red', 'red']):
    fig.add_hline(y=y_val, line_dash="dash", line_color=color, row=1, col=1, secondary_y=True, opacity=0.5)

# ----------------- 下層 (Row 2) -----------------
# 柱狀圖：大盤每日漲跌幅
fig.add_trace(
    go.Bar(x=df[date_col], y=df['taiex_pct'], name="大盤漲跌幅 (%)", marker_color=df['bar_color']),
    row=2, col=1, secondary_y=False
)

# ----------------- 版面與軸線設定 -----------------
fig.update_layout(
    title=f"台股加權指數 vs 市場寬度 (市值前 {len(tickers)} 大) 附漲跌幅動能",
    hovermode="x unified",
    plot_bgcolor='white',
    margin=dict(l=50, r=50, t=60, b=40),
)

# 套用至所有 X 軸：隱藏週末並加入時間縮放按鈕
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All")
        ])
    ),
    row=1, col=1
)

# 設定 Y 軸外觀
fig.update_yaxes(title_text="加權指數", row=1, col=1, secondary_y=False, showgrid=True, gridcolor='lightgray', zeroline=False)
fig.update_yaxes(title_text="市場寬度 (%)", range=[0, 100], row=1, col=1, secondary_y=True, showgrid=False, zeroline=False)
fig.update_yaxes(title_text="漲跌幅 (%)", row=2, col=1, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')

# 輸出瘦身版的 HTML
fig.write_html("index.html", auto_open=False, include_plotlyjs='cdn')
print("✅ 任務圓滿完成！圖表已儲存為 index.html")
