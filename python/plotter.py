"""this is the python script containing all of our plotting features"""

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import os

def plot_ticker(ticker):

    #open the csv for the indicated ticker
    df = pd.DataFrame()
    try:
        df = pd.read_csv('./dat/'+ticker+'.csv',parse_dates=True)
    except:
        print("[ERROR] could not open data for {}.".format(ticker))
        return

    ohlc = df.loc[:,['Date','Open','High','Low','Close','Volume']]
    #reindex by date
    ohlc = ohlc.set_index('Date')
    ohlc.index = pd.to_datetime(ohlc.index)
    
    if not os.path.exists('./plots'):
        os.system('mkdir ./plots/')

    mpf.plot(ohlc, type='candle',volume=True,savefig='./plots/'+ticker+'_candleWithVolume.png')


def plot_correlations(ticker):
    
    #open the csv for the indicated ticker
    df = pd.DataFrame()
    try:
        df = pd.read_csv('./dat/'+ticker+'.csv')
    except:
        print("[ERROR] could not open data for {}.".format(ticker))
        return

