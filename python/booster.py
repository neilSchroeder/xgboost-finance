import xgboost as xgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

import python.indicators as indicators

def add_metrics_regressor(data,ticker):
    """ adds metrics for use in the regression """
    data[ticker+' macd'] = indicators.macd(data[ticker+' Adj Close'].values, 12, 26, 9)
    data[ticker+' rsi'] = indicators.rsi(data[ticker+' Adj Close'].values, 14) #calculates RSI using 14 day window
    data['target'] = indicators.pct_diff(data[ticker+' Adj Close'].values, 2) #pct diff between day and day+X
    return data

def regress(ticker):
    """builds a regressor using xgboost to predict the 5 day pct diff for a given ticker"""

    data = pd.read_csv('dat/sp500_adj_close.csv')

    #add our additional metrics and target to our data
    if 'target' not in data.columns:
        data = add_metrics_regressor(data,ticker)

    #before going forward, drop the date from our data
    data = data.drop('Date',axis=1)
    X, y = data.loc[:, data.columns != 'target'], data['target']

    #make a sliding window of 3 weeks
    window = 21
    temp = [X]
    for i in range(window):
        right = X.shift(i+1)
        new_names = {col:col+f' {i+1}' for col in right.columns}
        right.rename(columns=new_names, inplace=True)
        temp.append(right)

    X = pd.concat(temp, axis=1)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(X.values,y.values,test_size=0.20)
    xg_regressor = xgb.XGBRegressor(use_label_encoders=False)
    xg_regressor.fit(x_train,y_train)

    # feature importances
    feat_import = pd.DataFrame(xg_regressor.feature_importances_.reshape(1,-1),columns=X.columns)
    feat_import.sort_values(0, axis=1, ascending=False, inplace=True)
    with open("feature_importances_"+ticker+".tsv",'w') as f:
        for col in feat_import:
            f.write(f'{col}\t{feat_import[col]}\n')

    #predict the output for the test values
    xg_predict = xg_regressor.predict(x_test)

    #check how well we did
    mask = y_test != 0
    y_test = y_test[mask]
    xg_predict = xg_predict[mask]
    explained_variance = 1 - np.divide(np.var(np.subtract(y_test, xg_predict)), y_test)
    rms_err = np.sqrt(mean_squared_error(xg_predict,y_test))
    chi_sqr = np.sum( np.divide(np.power(np.subtract(xg_predict, y_test),2), y_test))
    print(f'there are {len(y_test)} events in the test sample')
    print(f'the mean squared error is: {round(rms_err, 4)}')
    print(f'the reduced chi squared value is {chi_sqr/(len(y_test)-1)}')

    #plot the information
    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(xg_predict, label="predicted values")
    axs[0].plot(y_test, label="true values")
    axs[0].legend(loc='best')
    axs[0].set_ylabel(f'x day pct change for {ticker}')
    axs[1].plot(explained_variance, label="explained variance")
    axs[1].legend(loc='best')
    axs[1].set_ylabel("explained variance")
    plt.show()
