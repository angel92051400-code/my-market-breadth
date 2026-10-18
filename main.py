import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# 1. 這裡放置你抓取當日報價、計算 20MA/60MA 與市場寬度的邏輯
# 假設你把歷史資料存在 GitHub 裡的 history_data.csv，每天更新這份檔案
# df = pd.read_csv('history_data.csv')
# df = update_today_data(df) 
# df.to_csv('history_data.csv', index=False) # 存回最新資料

# --- 以下為模擬資料測試用，實戰請替換為你的 df ---
dates = pd.date_range(start="2023-01-01", end=datetime.date.today(), freq='B')
df = pd.DataFrame({
    'date': dates,
    'taiex': [15000 + i*10 for i in range(len(dates))], # 模擬加權指數
    'breadth': [40 + (i%30) for i in range(len(dates))] # 模擬市場寬度
})
# ----------------------------------------------

# 2. 建立雙 Y 軸圖表
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 加入加權指數 (主 Y 軸，左側)
fig.add_trace(
    go.Scatter(x=df['date'], y=df['taiex'], name="加權指數", mode='lines', line=dict(color='blue')),
    secondary_y=False,
)

# 加入市場寬度 (副 Y 軸，右側)
fig.add_trace(
    go.Scatter(x=df['date'], y=df['breadth'], name="市場寬度 (%)", mode='lines', line=dict(color='orange')),
    secondary_y=True,
)

# 3. 設定水平參考線 (20, 30, 50, 60)
for y_val, color in zip([20, 30, 50, 60], ['green', 'green', 'red', 'red']):
    fig.add_hline(y=y_val, line_dash="dash", line_color=color, secondary_y=True)

# 4. 版面細節與互動設定
fig.update_layout(
    title="台股加權指數 vs 站上月季線比例 (市場寬度)",
    hovermode="x unified", # 游標十字線對齊
    xaxis=dict(
        rangebreaks=[dict(bounds=["sat", "mon"])], # 隱藏週末空白
        rangeselector=dict( # 時間縮放按鈕
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
fig.update_yaxes(title_text="加權指數", secondary_y=False)
fig.update_yaxes(title_text="市場寬度 (%)", range=[0, 100], secondary_y=True)

# 5. 將圖表輸出為獨立的 HTML 網頁檔！(這是關鍵)
fig.write_html("index.html", auto_open=False)
print("網頁更新完成：index.html")
