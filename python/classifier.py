import xgboost as xgb
import pandas as pd

import indicators

def add_metrics(data,ticker):

    data[ticker+' macd'] = indicators.macd(data[ticker+' Adj Close'].values, 12, 26, 9)
    data[ticker+' rsi'] = indicators.rse(data[ticker+' Adj Close'].values, 14)
    #data[ticker+' stoch fast'] = indicators.stochastics_fast(data[ticker+' Adj Close'].values, 14, 3)
    data[ticker+' pct diff'] = indicators.pct_diff(data[ticker+' Adj Close'].values, 5) 
    data['target'] = indicators.build_target(data[ticker+' pct diff'].values, 3)
    return data

def analyze(ticker):
    """builds a classifier using xgboost to determine a buy, sell, hold strategy for a given ticker"""

    data = pd.read_csv('dat/sp500_adj_close.csv')

    #add our additional metrics and target to our data
    data = add_metrics(data,ticker)
