import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

print("啟動量化引擎：開始下載資料與計算市場寬度...")

# ==========================================
# 1. 取得台股加權指數與計算布林通道
# ==========================================
print("📥 步驟 1: 正在下載大盤加權指數 (^TWII) 並計算布林通道...")
taiex_data = yf.download("^TWII", period="2y", progress=False)
taiex_close = taiex_data['Close'].ffill() 
taiex_pct = taiex_close.pct_change() * 100

# 計算大盤布林通道 (20MA, 加減 2 倍標準差)
taiex_20ma = taiex_close.rolling(window=20).mean()
taiex_20std = taiex_close.rolling(window=20).std()
taiex_bb_upper = taiex_20ma + (2 * taiex_20std)
taiex_bb_lower = taiex_20ma - (2 * taiex_20std)

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
# 3. 批次下載成分股歷史報價
# ==========================================
print("📥 步驟 3: 正在從 Yahoo Finance 批次下載個股資料...")
stock_data = yf.download(tickers, period="2y", threads=True, progress=False)

if isinstance(stock_data.columns, pd.MultiIndex):
    close_prices = stock_data['Close']
else:
    close_prices = pd.DataFrame(stock_data['Close'])

# 向前填補 (Forward Fill)，防止因為單日資料遺失導致均線斷裂
close_prices = close_prices.ffill()

# ==========================================
# 4. 核心量化邏輯與修復
# ==========================================
print("🧮 步驟 4: 進行均線與市場寬度極速運算...")
ma20 = close_prices.rolling(window=20).mean()
ma60 = close_prices.rolling(window=60).mean()

above_both = (close_prices > ma20) & (close_prices > ma60)
breadth_series = above_both.mean(axis=1) * 100

# 將一開始完全算不出季線的「助跑期」設為空值
breadth_series[ma60.isna().all(axis=1)] = pd.NA

# ==========================================
# 5. 資料合併與清理
# ==========================================
print("🧹 步驟 5: 對齊時間軸與整理漲跌色彩...")
df = pd.DataFrame({
    'taiex': taiex_close.squeeze(),
    'taiex_bb_upper': taiex_bb_upper.squeeze(),
    'taiex_bb_lower': taiex_bb_lower.squeeze(),
    'taiex_pct': taiex_pct.squeeze(),
    'breadth': breadth_series.squeeze()
})

# 剔除所有的空值日 (包含前 60 天的助跑期)
df = df.dropna()
df = df.reset_index()
date_col = df.columns[0]

# 設定柱狀圖顏色 (漲=紅色，跌=綠色)
df['bar_color'] = np.where(df['taiex_pct'] >= 0, 'red', 'green')

# ==========================================
# 6. Plotly 雙層互動圖表生成
# ==========================================
print("📊 步驟 6: 正在生成子圖疊加 HTML 互動圖表...")

fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    row_heights=[0.75, 0.25],
    vertical_spacing=0.08,
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
)

# ----------------- 上層 (Row 1) -----------------
# Trace 0: 加權指數
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['taiex'], name="加權指數", mode='lines', line=dict(color='blue', width=2)),
    row=1, col=1, secondary_y=False
)

# Trace 1: 布林上軌 (預設隱藏)
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['taiex_bb_upper'], name="布林上軌", mode='lines', 
               line=dict(color='rgba(128, 128, 128, 0.5)', width=1, dash='dash'), 
               visible=False),
    row=1, col=1, secondary_y=False
)

# Trace 2: 布林下軌，並帶有半透明填滿效果 (預設隱藏)
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['taiex_bb_lower'], name="布林下軌", mode='lines', 
               fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)', 
               line=dict(color='rgba(128, 128, 128, 0.5)', width=1, dash='dash'), 
               visible=False),
    row=1, col=1, secondary_y=False
)

# Trace 3: 市場寬度
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['breadth'], name="市場寬度 (%)", mode='lines', line=dict(color='orange', width=2)),
    row=1, col=1, secondary_y=True
)

for y_val, color in zip([20, 30, 50, 60], ['green', 'green', 'red', 'red']):
    fig.add_hline(y=y_val, line_dash="dash", line_color=color, row=1, col=1, secondary_y=True, opacity=0.5)

# ----------------- 下層 (Row 2) -----------------
# Trace 4: 大盤漲跌幅
fig.add_trace(
    go.Bar(x=df[date_col], y=df['taiex_pct'], name="大盤漲跌幅 (%)", marker_color=df['bar_color']),
    row=2, col=1, secondary_y=False
)

# ----------------- 版面與互動按鈕設定 -----------------
fig.update_layout(
    title=dict(
        text=f"台股加權指數 vs 市場寬度 (市值前 {len(tickers)} 大) 附漲跌幅動能",
        x=0.5,             
        xanchor='center'   
    ),
    hovermode="x unified",
    plot_bgcolor='white',
    margin=dict(l=50, r=50, t=80, b=40),
    
    # 布林通道切換按鈕
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=1.0,         
            xanchor="right",
            y=1.08,        
            yanchor="bottom",
            showactive=True,
            buttons=list([
                dict(
                    label="隱藏布林通道",
                    method="restyle",
                    args=[{"visible": [True, False, False, True, True]}]
                ),
                dict(
                    label="顯示布林通道",
                    method="restyle",
                    args=[{"visible": [True, True, True, True, True]}]
                )
            ])
        )
    ]
)

# 👇👇👇 關鍵修復：將十字線設定「獨立」出來，套用到所有 X 軸 (不加 row/col)
fig.update_xaxes(
    rangebreaks=[dict(bounds=["sat", "mon"])],
    showspikes=True,       
    spikemode="across",    # 貫穿全域副圖
    spikesnap="cursor",    
    spikedash="solid",     
    spikethickness=1,      
    spikecolor="gray"
)

# 👇👇👇 時間縮放按鈕單獨設定，只綁定在第一張主圖上
fig.update_xaxes(
    rangeselector=dict(
        y=1.05, 
        x=0,
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

# Y 軸全域套用水平游標線
fig.update_yaxes(
    showspikes=True, 
    spikemode="across", 
    spikesnap="cursor", 
    spikedash="dot", 
    spikethickness=1, 
    spikecolor="gray"
)

fig.update_yaxes(title_text="加權指數", row=1, col=1, secondary_y=False, showgrid=True, gridcolor='lightgray', zeroline=False)
fig.update_yaxes(title_text="市場寬度 (%)", range=[0, 100], row=1, col=1, secondary_y=True, showgrid=False, zeroline=False)
fig.update_yaxes(title_text="漲跌幅 (%)", row=2, col=1, showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')

fig.write_html("index.html", auto_open=False, include_plotlyjs='cdn')
print("✅ 任務圓滿完成！圖表已儲存為 index.html")
