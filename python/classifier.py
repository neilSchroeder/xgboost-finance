import xgboost as xgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

import indicators

def add_metrics(data,ticker):

    data[ticker+' macd'] = indicators.macd(data[ticker+' Adj Close'].values, 12, 26, 9) 
    data[ticker+' rsi'] = indicators.rse(data[ticker+' Adj Close'].values, 14) #calculates RSI using 14 day window
    data[ticker+' pct diff'] = indicators.pct_diff(data[ticker+' Adj Close'].values, 5) #pct diff between day and day+5
    data['target'] = indicators.build_target(data[ticker+' pct diff'].values, 3) #target built on 3% difference in price over pct diff
    return data

def analyze(ticker):
    """builds a classifier using xgboost to determine a buy, sell, hold strategy for a given ticker"""

    data = pd.read_csv('dat/sp500_adj_close.csv')

    #add our additional metrics and target to our data
    if 'target' in data.columns:
        data = add_metrics(data,ticker)

    #before going forward, drop the date from our data
    data = data.drop('Date',axis=1)
    X, y = data.iloc[:,:-1], data.iloc[:,-1]
    #data_dmatrix = xgb.DMatrix(data=X,label=y)

    #need to split as time series so we can learn something about past information
    #this particular feature is likely to be the most interesting thing to play with for model stability and performance.
    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=28, test_size=7)
    splits = tscv.split(X.values)

    accuracy = []

    for train_index, test_index in splits:
        x_train, x_test = X.iloc[train_index].values, X.iloc[test_index].values
        y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

        xg_classifier = xgb.XGBClassifier(use_label_encoders=False)
        xg_classifier.fit(x_train,y_train)
        xg_predict = xg_classifier.predict(x_test)
        predictions = [round for x in xg_predict]
        accuracy.append(100*accuracy_score(y_test, predictions))
        print(xg_classifier, accuracy[-1])
