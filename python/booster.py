import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

import python.indicators as indicators
from python.tools import hyper_param_optimize as hpo
from python.tools import what_change

cut_sell = 0.1
cut_buy = 0.9

def add_metrics_regressor(data,ticker):
    """ adds metrics for use in the regression """
    data[ticker+' macd'] = indicators.macd(data[ticker+' Adj Close'].values, 12, 26, 9)
    data[ticker+' rsi'] = indicators.rsi(data[ticker+' Adj Close'].values, 14) #calculates RSI using 14 day window
    data['target'] = indicators.pct_diff(data[ticker+' Adj Close'].values, 5) #pct diff between day and day+X
    return data

def get_data(ticker):
    # gets the data for a particular ticker

    data = pd.read_csv('dat/sp500_adj_close.csv')

    # add our additional metrics and target to our data
    if 'target' not in data.columns:
        data = add_metrics_regressor(data,ticker)

    # before going forward, drop the date from our data
    dates = data['Date'].values
    data = data.drop('Date',axis=1)
    X = data.drop('target',axis=1)
    print(f"The maximum and minimum target values were {data['target'].max()}% and {data['target'].min()}% before normalizing")
    y = (data['target'] + abs(data['target'].min()))/(data['target'].max()-data['target'].min())
    global cut_sell
    global cut_buy
    median = np.median(y)
    width = np.std(y)
    cut_sell = median - width
    cut_buy = median + width

    # make a sliding window of 3 weeks
    window = 21
    temp = [X]
    for i in range(window):
        right = X.shift(i+1)
        new_names = {col:col+f' {i+1}' for col in right.columns}
        right.rename(columns=new_names, inplace=True)
        temp.append(right)

    X = pd.concat(temp, axis=1)

    X.fillna(0,inplace=True)
    y.fillna(0,inplace=True)

    return X, y

def tsne_decompose(x, y_actions,ticker):
    # performs TSNE decomposition on your data
    actions = ["sell", "hold", "buy"]
    colors = ['dodgerblue', 'limegreen', 'crimson']

    fig = plt.figure()
    axs = fig.add_subplot()
    ret1 = []
    ret2 = []

    tsne_train = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=3000)
    tsne_train_results = tsne_train.fit_transform(x)

    for act in actions:
        mask = [x == act for x in y_actions]
        tsne_act = tsne_train_results[mask]
        axs.scatter(tsne_act[:,0],
                tsne_act[:,1],
                color=colors[actions.index(act)],
                marker='.',
                alpha=0.75,
                label=act)


    axs.set_xlabel("T-SNE var 1")
    axs.set_ylabel("T-SNE var 2")
    axs.legend(loc='best')
    fig.savefig(f"./plots/tsne_decomp_{ticker}.png")

    return tsne_train_results[:,0], tsne_train_results[:,1]

def split(x,y, pct):
    #returns split data
    split_index = int(pct*len(y))
    return x[:split_index], x[split_index::], y[:split_index], y[split_index::]

def xg_forecast(x, y, x_test, ticker):
    # trains a forecast model using xgboost

    # create regression ready dataset
    dm = xgb.DMatrix(data=x, label=y)

    # these were determined by the bayesian optimization procedure
    params = {"objective":'reg:squaredlogerror',
                "eval_metric":'rmse',
                'colsample_bytree': 0.4182626970417747, 
                'gamma': 0.0008409788767502647, 
                'learning_rate': 0.38790242633847205, 
                'max_depth': int(8.530158311303914), 
                'min_child_weight': 4.593283204320386, 
                'num_boost_rounds': 8.106460483344513, 
                'subsample': 0.8099971184702701}

    # train the xgboost model
    xg_m = xgb.train(params=params, dtrain=dm)

    # predict the output for the test values
    dm_test = xgb.DMatrix(data=x_test)
    xg_predict = xg_m.predict(dm_test)
    xg_m.dump_model(f"models/model_{ticker}.txt")

    return xg_predict

def walk_forward(x_train, x_test, y_train, y_test, ticker):
    # walk forward validation
    global cut_buy
    global cut_sell

    predictions = []
    history_x = [x for x in x_train]
    history_y = [y for y in y_train]


    for i in range(len(x_test)):

        prediction = xg_forecast(history_x, history_y, x_test, ticker)

        predictions.append(prediction)
        history_x.append(x_test[i])
        history_y.append(y_test[i])

        print(f'expected={round(history_y[-1],4)}, predicted={round(prediction,4)}')
        print(f'expected act = {what_change([history_y[-1]], cut_sell, cut_buy)}, predicted act={what_change([prediction],cut_sell, cut_buy)} ')

    error = mean_squared_error(y_test, predictions)
    return error, predictions


def regress(args):
    """builds a regressor using xgboost to predict the 5 day pct diff for a given ticker"""
    global cut_buy
    global cut_sell

    X, y = get_data(args.ticker)

    # tsne decomposition
    if args.tsne:
        # highly dimensional datasets need to be reduced
        X_norm = (X - X.mean())/X.std()
        X_norm.fillna(0,inplace=True)
        pca = PCA(n_components=50)
        pca_result_50 = pca.fit_transform(X_norm.values)

        #for i in range(5):
        #    df = pd.DataFrame({f'PC{i}':pca.components_[i], 'Variable Names':list(X_norm.columns)})
        #    df.sort_values(f'PC{i}', ascending=False, inplace=True)
        #    print('The top 10 most important variables are')
        #    print(df.head(50))

        loadings = pd.DataFrame(pca_result_50,
                                columns=[f'PC_{x}' for x in range(50)],
                                index=X.index)
        
        print('Cumulative explained variation for 50 principle components')
        print(f'{np.sum(pca.explained_variance_ratio_)}')

        # apply actions to y_vals
        y_actions = what_change(y.values, cut_sell, cut_buy)

        # TNSE decomposition:
        var1, var2 = tsne_decompose(pca_result_50, y_actions, args.ticker)
    
        loadings["tsne 1"] = var1
        loadings["tsne 2"] = var2
        
        X = loadings

    # train test split for walk-forward 
    x_train, x_test, y_train, y_test = [], [], [], []

    # walk-forward validation
    xg_predict = []
    if args.walk:
        x_train, x_test, y_train, y_test = split(X.values, y.values, args.split)
        err, xg_predict = walk_forward(x_train, x_test, y_train, y_test, args.ticker)
        # check how well we did
        print(f"the mean squared error is: {err}")
    else:
        x_train, x_test, y_train, y_test = train_test_split(X.values, y.values, test_size=args.split)
        if args.hpo:
            params = hpo(x_train, y_train, cut_sell, cut_buy)
        xg_predict = xg_forecast(x_train, y_train, x_test, args.ticker)

    # mean squared error
    print(f'the mean squared error is {mean_squared_error(y_test,xg_predict)}')
    print(f'the accuracy score is {accuracy_score(what_change(y_test, cut_sell, cut_buy),what_change(xg_predict, cut_sell, cut_buy))}')

    if args.ratio:
        # plot the information
        actions = ["sell", "hold", "buy"]
        colors = ['dodgerblue', 'limegreen', 'crimson']
        fig, axs = plt.subplots(nrows=2, ncols=len(actions))

        for i,act in enumerate(actions):
            action_mask = [x == act for x in what_change(y_test, cut_sell, cut_buy)]
            y_vals = y_test[action_mask]
            x_vals = xg_predict[action_mask]
            y_actions = what_change(y_vals, cut_sell, cut_buy)
            x_actions = what_change(x_vals, cut_sell, cut_buy)
            explained_variance = [actions.index(y_actions[j]) - actions.index(x_actions[j]) for j in range(len(y_actions))]
            axs[0][i].plot(y_vals, label="true values", marker='.', linewidth=0)
            axs[0][i].plot(x_vals, label="predicted values", marker='.', linewidth=0)
            axs[0][i].legend(loc='best')
            axs[0][i].set_ylabel(f'x day pct change for {args.ticker}')
            axs[0][i].set_title(f"Results for {act}")
            axs[1][i].plot(explained_variance, label="True Action - Predicted Action", marker='.', linewidth=0)
            axs[1][i].set_ylabel("True Action - Predicted Action")
    
        plt.show()

