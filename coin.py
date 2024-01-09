# 참고
# https://www.youtube.com/watch?v=JLVB8ZUPojw&list=PLXKfqSKO-IvNFOm-y7GOM5LWnBC1h1xuS&index=9&t=30s

import streamlit as st
from cryptocmd import CmcScraper
import plotly.graph_objects as go
from datetime import datetime
from prophet import Prophet
import requests
import json

st.write('''
# :money_mouth_face: 코인
코인 그래프와 예측 그래프(Predicted Close)
''')
# https://coinmarketcap.com
st.sidebar.header('Menu')

# 코인 이름
# 75259f98-8e3e-453a-813c-49b6bf24375c
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': st.secrets['coin_api_key'],
}

response = requests.get(url, headers=headers)
data = json.loads(response.text)

coins = list()

for coin in data['data']:
    coins.append(coin['symbol'])
############################################################################

name = st.sidebar.selectbox('Name', coins)

start_date = st.sidebar.date_input('Start date', datetime(2024, 1, 1))
end_date = st.sidebar.date_input('End date', datetime(2024,1,7))

scraper = CmcScraper(name,start_date.strftime('%d-%m-%Y'), end_date.strftime('%d-%m-%Y'))
df = scraper.get_dataframe()
# 변수
# Date, Open, High, Low, Close - 종가, Volume, Market Cap, Time Open, Time High, Time Low, Time Close

fig_close = go.Figure(data=[go.Scatter(x=df['Date'], y=df[col], name=col) for col in ['Open', 'High', 'Low', 'Close']],
                      layout=go.Layout(title='가격'))
fig_volume = go.Figure(data = [go.Scatter(x=df['Date'], y=df['Volume'], name = 'Volume')], layout=go.Layout(title='거래량'))

# 예측 그래프
df_prophet = df[['Date','Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=5) # set day
forecast = model.predict(future)
fig_close.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Close'))
###########################################################################################################################

st.plotly_chart(fig_close)
st.plotly_chart(fig_volume)
