# download_stock_data.py

import yfinance as yf
import pandas as pd

tickers = ['JPM', 'XOM', 'NVDA', 'NAS.OL', 'META']

daily_data = yf.download(tickers, period='1y', interval='1d')
daily_data.to_csv('daily_stock_data.csv')

weekly_data = yf.download(tickers, period='2y', interval='1wk')
weekly_data.to_csv('weekly_stock_data.csv')

