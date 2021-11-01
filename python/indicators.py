import pandas as pd
import numpy as np


def get_ema(data, days):
    """calculates the exponenetial moving average for a set of data over a period given by days"""
    weights = np.exp(np.linspace(-1,0.,days))
    weights /= weights.sum() #normalize the weights

    z = np.convolve(data,weights)[:len(data)] #fold the weights into the data to calculate the ema
    z[:days] = z[days] #ensure the ema has the same length as the data
    return z


def macd(data, fast, slow, signal):
    """calculates the macd for a ticker 
    fast is the exponential moving average in days for the fast signal
    slow is the exponential moving average in days for the slow signal
    signal is the exponential moving average in days of the difference of the fast and slow signals"""
    data = np.array(data)
    ema_fast = get_ema(data, fast)
    ema_slow = get_ema(data, slow)
    ema_signal = get_ema(np.subtract(ema_fast,ema_slow), signal)

    return ema_signal


def rsi(data, days):
    """calculates the relative strength index for a ticker over a period of days"""
    
    data = np.array(data)

    diff = np.diff(data,n=1,prepend=data[0])
    up_vals = diff > 0
    down_vals = diff < 0

    up_diff = np.multiply(diff,up_vals)
    down_diff = np.multiply(diff,down_vals)

    avg_up_diff = np.convolve(up_diff, np.ones(days))[:len(up_diff)] / days
    avg_up_diff[:days] = avg_up_diff[days]
    avg_down_diff = np.convolve(down_diff, -1*np.ones(days))[:len(down_diff)] / days
    avg_down_diff[:days] = avg_down_diff[days]

    ret = np.array([100 - (100 / ( 1 + ((avg_up_diff[i-1]*(days-1)+avg_up_diff[i])/(avg_down_diff[i-1]*(days-1)+avg_down_diff[i])))) if i >= 14 else 100 - (100 / ( 1 + (avg_up_diff[i]/avg_down_diff[i]))) for i in range(len(avg_up_diff))])
    return ret
    

def pct_diff(data, days):
    """calculates the percent difference between a day and x days into the future"""
    return [100*(data[i+days] - data[i])/data[i] if i < len(data)-days else 100*(data[-1] - data[i])/data[i] for i in range(len(data))]


def build_target(data, cut):
    """returns the buy/sell/hold classification based on the x day percent diff and a manually set cut (I'll use 5% for now)"""
    ret = np.array([int(np.sign(data[i])) if abs(data[i]) >= cut else 0 for i in range(len(data))])
    print("there are {} buy days".format(np.sum(ret == 1)))
    print("there are {} sell days".format(np.sum(ret == -1)))
    print("there are {} hold days".format(np.sum(ret == 0)))
    return ret
