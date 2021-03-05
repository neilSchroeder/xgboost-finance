"""
This is the scraper.

This script will pull the s&p500 tickers from wikipedia and then pull the data for these companies from IEX's finance API
"""

import datetime
import numpy as np
import os
import pandas as pd
from pandas_datareader import data as pdr
import requests
import yfinance as yf
yf.pdr_override()

def get_sAndP500tickers():
    """uses pandas to scrape the s&p500 wiki for the ticker list"""
    table = pd.read_html('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv('dat/s_and_p_500_tickers.tsv', sep='\t', columns=['Symbol'])

def get_tickerData():
    """uses pandas and iexfinance to get historical daily data for the s&p500"""
    #get tickers
    tickers = pd.read_csv('dat/s_and_p_500_tickers.tsv', delimiter='\t', index_col=[0])

    start = datetime.date(2010,1,1)
    end = datetime.date.today() - datetime.timedelta(days=1)

    #yahoo is not kind about this data, so I'm going to find a workaround
    
    for ticker in tickers['Symbol'].values:
        print(ticker)
        data=pdr.get_data_yahoo(ticker, start=start, end=end)
        print(data.head())
        data.to_csv('dat/'+ticker+'.csv')


def get_data():
    """gets the tickers if necessary, and updates the ticker data in the range of 2013 to today"""
    if not os.path.exists('./dat/s_and_p_500_tickers.tsv'):
        print('check')
        get_sAndP500tickers()

    get_tickerData()

