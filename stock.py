import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
from cryptocmd import CmcScraper
from datetime import datetime

# '005930.KS'(삼성 종목 코드)의 주식 데이터
# https://finance.yahoo.com/quote/005930.KS?p=005930.KS
# https://finance.yahoo.com/quote/131970.KS?p=131970.KS
# 몇몇은 .KQ
## 주식정보 받아오기
def get_krx_stock_codes():
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    df = pd.read_html(url, header=0, encoding='CP949')[0]
    df.종목코드 = df.종목코드.map('{:06d}'.format) + '.KS'  
    return df

df = get_krx_stock_codes().dropna()

selected_name = st.sidebar.selectbox('Name', df['회사명'])
selected_code = df[df['회사명'] == selected_name]['종목코드'].values[0]
########################################################################################

st.write(f'''
# {selected_name} 주식 데이터
마감 가격과 거래량 시각화, 예측모델
''')

### testing zone
start_date = st.sidebar.date_input('Start date ( Recommend at least 6 months )', datetime(2024, 1, 1))
end_date = st.sidebar.date_input('End date', datetime(2024,1,7))
period = st.sidebar.number_input('Forecast period', min_value=1, value=5, step=1)
scraper = CmcScraper(selected_name,start_date.strftime('%d-%m-%Y'), end_date.strftime('%d-%m-%Y'))
df = yf.download(selected_code, start=start_date, end=end_date)
#################
# df = yf.download(selected_code, start='2023-08-01', end='2024-01-07')

if len(df) < 2:
    st.error(f'{selected_name}의 데이터가 없습니다. 다른 종목을 선택해주세요.')
else:

    # 마감가격
    fig1 = go.Figure(data = [go.Scatter(x=df.index, y=df['Close'], name = 'Close Price')], 
                    layout=go.Layout(title='마감 가격', xaxis_title='날짜', yaxis_title='가격'))

    # 거래량
    fig2 = go.Figure(data = [go.Scatter(x=df.index, y=df['Volume'], name = 'Volume')], 
                    layout=go.Layout(title='거래량', xaxis_title='날짜', yaxis_title='거래량'))

    ## 예측 그래프
    df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=5) # set day
    forecast = model.predict(future)
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Close'))
    ###########################################################################################################################

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)