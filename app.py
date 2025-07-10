import streamlit as st
import yfinance as yf
from stocker.stocker import Stocker
import matplotlib.pyplot as plt

# 網頁標題
st.title("📈 股票預測系統 (Powered by Prophet)")

# 股票代碼輸入
symbol = st.text_input("輸入股票代碼（如 2330.TW）", value="2330.TW")

# 預測天數
days = st.slider("預測未來天數", min_value=7, max_value=180, value=30, step=7)

# 開始預測按鈕
if st.button("開始預測"):
    with st.spinner("正在載入資料與訓練模型..."):
        # 抓資料
        df = yf.download(symbol, start="2020-01-01")
        series = df["Close"]
        series.name = symbol

        # 建立 Stocker 物件
        model = Stocker(series)

        # 預測
        _, future = model.create_prophet_model(days=days)

        # 顯示圖表
        st.pyplot(plt)
