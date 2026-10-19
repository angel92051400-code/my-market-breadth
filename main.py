import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

print("啟動量化引擎：開始下載資料與計算市場寬度...")

# ==========================================
# 1. 取得台股加權指數歷史資料 (過去 1 年)
# ==========================================
print("📥 步驟 1: 正在下載大盤加權指數 (^TWII)...")
taiex_data = yf.download("^TWII", period="1y", progress=False)
taiex_close = taiex_data['Close']

# ==========================================
# 2. 自動讀取 650 檔股票名單 (tickers.csv)
# ==========================================
csv_file = 'tickers.csv'
if os.path.exists(csv_file):
    print(f"📁 步驟 2: 找到 {csv_file}，正在讀取股票代號...")
    ticker_df = pd.read_csv(csv_file)
    
    # 防呆：確保讀到正確的欄位，沒有標題就硬抓第一欄
    if 'Symbol' in ticker_df.columns:
        tickers = ticker_df['Symbol'].astype(str).tolist()
    else:
        tickers = ticker_df.iloc[:, 0].astype(str).tolist()
else:
    print(f"⚠️ 找不到 {csv_file}！請確認檔案已上傳至 GitHub。")
    print("作為緊急備用，將使用預設的台灣前 10 大權值股名單...")
    tickers = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", 
               "2882.TW", "2891.TW", "3231.TW", "2382.TW", "2356.TW"]

# 清理代號，去除可能不小心按到的空白
tickers = [t.strip() for t in tickers if t.strip()]
print(f"🎯 準備抓取 {len(tickers)} 檔股票資料！")

# ==========================================
# 3. 批次下載成分股歷史報價
# ==========================================
print("📥 步驟 3: 正在從 Yahoo Finance 批次下載個股資料...")
# threads=True 會開啟多執行緒加速下載
stock_data = yf.download(tickers, period="1y", threads=True, progress=False)

# 處理 yfinance 回傳的多層次欄位格式
if isinstance(stock_data.columns, pd.MultiIndex):
    close_prices = stock_data['Close']
else:
    close_prices = pd.DataFrame(stock_data['Close'])

# ==========================================
# 4. 核心量化邏輯：Pandas 向量化計算市場寬度
# ==========================================
print("🧮 步驟 4: 進行均線與市場寬度極速運算...")
# 一次性算出所有股票的 20MA 與 60MA
ma20 = close_prices.rolling(window=20).mean()
ma60 = close_prices.rolling(window=60).mean()

# 判斷是否「同時站上」月線與季線
above_both = (close_prices > ma20) & (close_prices > ma60)

# 計算每日符合條件的比例 (%)
breadth_series = above_both.mean(axis=1) * 100

# ==========================================
# 5. 資料合併與清理
# ==========================================
print("🧹 步驟 5: 對齊大盤與寬度時間軸...")
df = pd.DataFrame({
    'taiex': taiex_close.squeeze(),
    'breadth': breadth_series.squeeze()
})

# 剔除前 60 天均線還沒算出來的空白(NaN)日子
df = df.dropna()
df = df.reset_index()
# yfinance 的日期欄位名稱通常為 Date 或 index
date_col = df.columns[0]

# ==========================================
# 6. Plotly 互動圖表生成
# ==========================================
print("📊 步驟 6: 正在生成 HTML 互動圖表...")
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 藍線：加權指數
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['taiex'], name="加權指數", mode='lines', line=dict(color='blue', width=2)),
    secondary_y=False,
)

# 橘線：市場寬度
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['breadth'], name="市場寬度 (%)", mode='lines', line=dict(color='orange', width=2)),
    secondary_y=True,
)

# 紅綠警戒線
for y_val, color in zip([20, 30, 50, 60], ['green', 'green', 'red', 'red']):
    fig.add_hline(y=y_val, line_dash="dash", line_color=color, secondary_y=True, opacity=0.5)

# 圖表外觀與互動設定
fig.update_layout(
    # 這裡的標題會自動抓取名單的數量，如果有讀到 tickers.csv，就會顯示 650 大
    title=f"台股加權指數 vs 市場寬度 (市值前 {len(tickers)} 大)",
    hovermode="x unified",
    plot_bgcolor='white',
    xaxis=dict(
        rangebreaks=[dict(bounds=["sat", "mon"])], # 隱藏週末
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
)

fig.update_yaxes(title_text="加權指數", secondary_y=False, showgrid=True, gridcolor='lightgray', zeroline=False)
fig.update_yaxes(title_text="市場寬度 (%)", range=[0, 100], secondary_y=True, showgrid=False, zeroline=False)

# 輸出瘦身版的 HTML (套用 CDN)
fig.write_html("index.html", auto_open=False, include_plotlyjs='cdn')
print("✅ 任務圓滿完成！圖表已儲存為 index.html")
