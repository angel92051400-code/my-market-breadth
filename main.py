import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("開始執行：下載資料與計算指標...")

# ==========================================
# 1. 取得台股加權指數歷史資料 (過去 1 年)
# ==========================================
print("1. 正在下載加權指數...")
taiex_data = yf.download("^TWII", period="1y")
# yfinance 下載下來的資料是 DataFrame，我們取出收盤價 (Close)
taiex_close = taiex_data['Close']

# ==========================================
# 2. 定義你的股票池 (這裡以台灣前 10 大權值股為例)
# ==========================================
# 💡 實戰擴充指南：
# 你可以在 GitHub 儲存庫裡建一個 tickers.csv，裡面放 650 家公司的代號
# 然後用 pd.read_csv('tickers.csv') 把清單讀進來。
# 注意：yfinance 的台股代號，上市要加 .TW，上櫃要加 .TWO
tickers = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", 
    "2882.TW", "2891.TW", "3231.TW", "2382.TW", "2356.TW"
]

# ==========================================
# 3. 批次下載成分股歷史報價
# ==========================================
print(f"2. 正在下載 {len(tickers)} 檔成分股資料...")
# yfinance 支援多執行緒，一次下載多檔股票速度很快
stock_data = yf.download(tickers, period="1y")

# 取出所有股票的收盤價矩陣 (直欄是股票代號，橫列是日期)
close_prices = stock_data['Close']

# ==========================================
# 4. 核心量化邏輯：Pandas 向量化計算市場寬度
# ==========================================
print("3. 進行均線與市場寬度運算...")
# 沿著時間軸向下滾動，計算 20MA 與 60MA
ma20 = close_prices.rolling(window=20).mean()
ma60 = close_prices.rolling(window=60).mean()

# 判斷每檔股票的每日收盤價，是否「同時大於」20MA 與 60MA
# 這會產生一個全都是 True / False 的矩陣
above_both = (close_prices > ma20) & (close_prices > ma60)

# 沿著股票代號的維度 (axis=1) 計算每天 True 的比例，並乘上 100 轉為百分比
breadth_series = above_both.mean(axis=1) * 100

# ==========================================
# 5. 資料合併與清理
# ==========================================
# 把加權指數和算好的市場寬度對齊合併
df = pd.DataFrame({
    'taiex': taiex_close.squeeze(),    # 確保是一維序列
    'breadth': breadth_series.squeeze()
})

# 因為前 60 天沒有 60MA 的數據（會是 NaN），所以要把前面沒有寬度資料的日子刪除
df = df.dropna()
# 將 index 轉為欄位，方便繪圖
df = df.reset_index()
# yfinance 下載的日期欄位名稱通常為 'Date'
date_col = 'Date' 

# ==========================================
# 6. Plotly 繪圖模組 (與原本幾乎相同)
# ==========================================
print("4. 正在生成 HTML 互動圖表...")
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 加入加權指數
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['taiex'], name="加權指數", mode='lines', line=dict(color='blue', width=2)),
    secondary_y=False,
)

# 加入市場寬度
fig.add_trace(
    go.Scatter(x=df[date_col], y=df['breadth'], name="市場寬度 (%)", mode='lines', line=dict(color='orange', width=2)),
    secondary_y=True,
)

# 設定水平參考線
for y_val, color in zip([20, 30, 50, 60], ['green', 'green', 'red', 'red']):
    fig.add_hline(y=y_val, line_dash="dash", line_color=color, secondary_y=True, opacity=0.5)

# 版面設定
fig.update_layout(
    title="台股加權指數 vs 站上月季線比例 (市場寬度)",
    hovermode="x unified",
    plot_bgcolor='white',
    xaxis=dict(
        rangebreaks=[dict(bounds=["sat", "mon"])],
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

fig.update_yaxes(title_text="加權指數", secondary_y=False, showgrid=True, gridcolor='lightgray')
fig.update_yaxes(title_text="市場寬度 (%)", range=[0, 100], secondary_y=True, showgrid=False)

# 輸出包含繪圖引擎的網頁，採用 CDN 縮小檔案體積
fig.write_html("index.html", auto_open=False, include_plotlyjs='cdn')
print("✅ 更新完成！已儲存為 index.html")
