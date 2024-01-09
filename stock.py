import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet

st.write('''
# 삼성전자 주식 데이터
마감 가격과 거래량 시각화
''')

# '005930.KS'(삼성 종목 코드)의 주식 데이터
# https://finance.yahoo.com/quote/005930.KS?p=005930.KS
# 두산 테스나 131970.KQ
df = yf.download('131970.KQ', start='2023-08-01', end='2024-01-07')

# 마감가격
fig1 = go.Figure(data = [go.Scatter(x=df.index, y=df['Close'], name = 'Close Price')], 
                 layout=go.Layout(title='마감 가격', xaxis_title='날짜', yaxis_title='가격'))

# 거래량
fig2 = go.Figure(data = [go.Scatter(x=df.index, y=df['Volume'], name = 'Volume')], 
                 layout=go.Layout(title='거래량', xaxis_title='날짜', yaxis_title='거래량'))

## 예측 그래프
# df_prophet = df[['Date','Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=5) # set day
forecast = model.predict(future)
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Close'))
###########################################################################################################################

st.plotly_chart(fig1)
st.plotly_chart(fig2)

# Easy