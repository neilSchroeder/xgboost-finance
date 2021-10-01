import xgboost as xgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

import indicators

def add_metrics_classifier(data,ticker):

    data[ticker+' macd'] = indicators.macd(data[ticker+' Adj Close'].values, 12, 26, 9)
    data[ticker+' rsi'] = indicators.rsi(data[ticker+' Adj Close'].values, 14) #calculates RSI using 14 day window
    data[ticker+' pct diff'] = indicators.pct_diff(data[ticker+' Adj Close'].values, 5) #pct diff between day and day+5
    data['target'] = indicators.build_target(data[ticker+' pct diff'].values, 3) #target built on 3% difference in price over pct diff
    return data

def add_metrics_regressor(data,ticker):

    data[ticker+' macd'] = indicators.macd(data[ticker+' Adj Close'].values, 12, 26, 9)
    data[ticker+' rsi'] = indicators.rsi(data[ticker+' Adj Close'].values, 14) #calculates RSI using 14 day window
    data['target'] = indicators.pct_diff(data[ticker+' Adj Close'].values, 7) #pct diff between day and day+X
    return data

def regress(ticker, split):
    """builds a regressor using xgboost to predict the 5 day pct diff for a given ticker"""

    data = pd.read_csv('dat/sp500_adj_close.csv')

    #add our additional metrics and target to our data
    if 'target' not in data.columns:
        data = add_metrics_regressor(data,ticker)

    #before going forward, drop the date from our data
    data = data.drop('Date',axis=1)
    X, y = data.loc[:, data.columns != 'target'], data['target']
    #make a sliding window of 1 week
    window = 21
    temp = [X]
    for i in range(window):
        right = X.shift(i+1)
        new_names = {col:col+f' {i+1}' for col in right.columns}
        right.rename(columns=new_names, inplace=True)
        temp.append(right)

    X = pd.concat(temp, axis=1)

    #data_dmatrix = xgb.DMatrix(data=X,label=y)


    if split == 'time_series':
        #need to split as time series so we can learn something about past information
        #this particular feature is likely to be the most interesting thing to play with for model stability and performance.
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=6)
        splits = tscv.split(X.values)

        accuracy = []
        for train_index, test_index in splits:
            x_train, x_test = X.iloc[train_index].values, X.iloc[test_index].values
            y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
            print("there are {} sequential entries in this training sample".format(len(train_index)))

            xg_regressor = xgb.XGBRegressor(use_label_encoders=False)
            xg_regressor.fit(x_train,y_train)
            xg_predict = xg_regressor.predict(x_test)
            truth = [xg_predict[i] == y_test[i] for i in range(len(y_test))]
            accuracy = 100 * np.sum([xg_predict[i] == y_test[i] for i in range(len(y_test))])/len(y_test)
            print("there were {} entries in the test sample".format(len(y_test)))
            print("there were {} correct guesses".format(np.sum(truth)))
            print("there were {} incorrect guesses".format(len(y_test)-np.sum(truth)))
            print("model performs with accuracy : {} %".format(accuracy))
            total = 1000.
            shares = 0
            dfy = y.iloc[test_index]
            for i,row in X.iloc[test_index].iterrows():
                if xg_predict[i-test_index[0]] == 1 and total > row[ticker+' Adj Close']:
                    shares += np.floor(total/row[ticker+' Adj Close'])
                    total -= np.floor(total/row[ticker+' Adj Close']) * row[ticker+' Adj Close']
                if xg_predict[i-test_index[0]] == -1 and shares > 0:
                    total += shares * row[ticker+' Adj Close']
                    shares = 0
                if i == test_index[-1] and shares != 0:
                    total += shares * row[ticker+' Adj Close']

            print("if you started with $1000, this model would make you ${}".format(total-1000))

    elif split == 'traditional':

        x_train, x_test, y_train, y_test = train_test_split(X.values,y.values,test_size=0.20)
        xg_regressor = xgb.XGBRegressor(use_label_encoders=False)
        xg_regressor.fit(x_train,y_train)
        feat_import = pd.DataFrame(xg_regressor.feature_importances_.reshape(1,-1),columns=X.columns)
        with open("feature_importances.tsv",'w') as f:
            for col in feat_import:
                f.write(f'{col}\t{feat_import[col]}\n')

        xg_predict = xg_regressor.predict(x_test)
        rms_err = np.sqrt(mean_squared_error(xg_predict,y_test))
        ratios_pct = np.abs(100*(1 - np.divide(y_test,xg_predict)))
        ratios_pct_mean = np.mean(ratios_pct)
        ratios_pct_stddev = np.std(ratios_pct)
        ratios_pct_distanceFromMean = np.abs(ratios_pct - ratios_pct_mean)
        not_outlier = ratios_pct_distanceFromMean < 5 * ratios_pct_stddev
        ratios_pct = ratios_pct[not_outlier]
        ratios_pct_mean = np.mean(ratios_pct)
        print("there were {} entries in the test sample".format(len(y_test)))
        print("model has RMS error: {}%".format(round(rms_err,4)))
        print("model, on average, predicts correct value to within {}% of the true value".format(round(ratios_pct_mean,4)))
        """
        bin_content, bin_edges = np.histogram(ratios_pct, bins=np.arange(0, 500, 10))
        plt.hist(bin_edges[:-1], bin_edges, weights=bin_content )
        plt.ylabel("Counts")
        plt.xlabel("Percent difference between true and predicted values")
        plt.title("Histogram of Percent Diff. for 9 Day Pct. Diff. Regressor")
        """

        fig, axs = plt.subplots(nrows=2, ncols=1)
        axs[0].plot(xg_predict, label="predicted values")
        axs[0].plot(y_test, label="true values")
        axs[0].legend(loc='best')
        axs[0].set_ylabel(f'1 day pct change for {ticker}')
        axs[1].plot(my_ratio, label="100*(true/predicted - 1)")
        axs[1].legend(loc='best')
        axs[1].set_ylabel("% difference")
        plt.show()

    else:
        print("[ERROR] split method {} not supported".format(split))
        return

def classify(ticker, split):
    """builds a classifier using xgboost to determine a buy, sell, hold strategy for a given ticker"""

    data = pd.read_csv('dat/sp500_adj_close.csv')

    #add our additional metrics and target to our data
    if 'target' not in data.columns:
        data = add_metrics_classifier(data,ticker)

    #before going forward, drop the date from our data
    data = data.drop('Date',axis=1)
    data = data.drop(ticker+' pct diff', axis=1)
    X, y = data.loc[:, data.columns != 'target'], data['target']
    #data_dmatrix = xgb.DMatrix(data=X,label=y)


    if split == 'time_series':
        #need to split as time series so we can learn something about past information
        #this particular feature is likely to be the most interesting thing to play with for model stability and performance.
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=6)
        splits = tscv.split(X.values)

        accuracy = []
        for train_index, test_index in splits:
            x_train, x_test = X.iloc[train_index].values, X.iloc[test_index].values
            y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
            print("there are {} sequential entries in this training sample".format(len(train_index)))

            xg_classifier = xgb.XGBClassifier(use_label_encoders=False)
            xg_classifier.fit(x_train,y_train)
            xg_predict = xg_classifier.predict(x_test)
            truth = [xg_predict[i] == y_test[i] for i in range(len(y_test))]
            accuracy = 100 * np.sum([xg_predict[i] == y_test[i] for i in range(len(y_test))])/len(y_test)
            print("there were {} entries in the test sample".format(len(y_test)))
            print("there were {} correct guesses".format(np.sum(truth)))
            print("there were {} incorrect guesses".format(len(y_test)-np.sum(truth)))
            print("model performs with accuracy : {} %".format(accuracy))
            total = 1000.
            shares = 0
            dfy = y.iloc[test_index]
            for i,row in X.iloc[test_index].iterrows():
                if xg_predict[i-test_index[0]] == 1 and total > row[ticker+' Adj Close']:
                    shares += np.floor(total/row[ticker+' Adj Close'])
                    total -= np.floor(total/row[ticker+' Adj Close']) * row[ticker+' Adj Close']
                if xg_predict[i-test_index[0]] == -1 and shares > 0:
                    total += shares * row[ticker+' Adj Close']
                    shares = 0
                if i == test_index[-1] and shares != 0:
                    total += shares * row[ticker+' Adj Close']

            print("if you started with $1000, this model would make you ${}".format(total-1000))

    elif split == 'traditional':

        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(X.values,y.values,test_size=0.20)
            xg_classifier = xgb.XGBClassifier(use_label_encoders=False)
            xg_classifier.fit(x_train,y_train)
            xg_predict = xg_classifier.predict(x_test)
            truth = [xg_predict[i] == y_test[i] for i in range(len(y_test))]
            accuracy = 100 * np.sum([xg_predict[i] == y_test[i] for i in range(len(y_test))])/len(y_test)
            print("there were {} entries in the test sample".format(len(y_test)))
            print("there were {} correct guesses".format(np.sum(truth)))
            print("there were {} incorrect guesses".format(len(y_test)-np.sum(truth)))
            print("model performs with accuracy : {} %".format(accuracy))

    else:
        print("[ERROR] split method {} not supported".format(split))
        return
