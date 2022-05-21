import xgboost as xgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

import python.indicators as indicators

def what_change(pct):
    # determines what action to take based on the pct change
    if pct < 0.448:
        return "hard sell"
    elif 0.448 < pct < 0.547:
        return "soft sell"
    elif 0.547 < pct < 0.5965:
        return "hold"
    elif 0.5965 < pct < 0.6955:
        return "soft buy"
    else:
        return "hard buy"
    

def add_metrics_regressor(data,ticker):
    """ adds metrics for use in the regression """
    data[ticker+' macd'] = indicators.macd(data[ticker+' Adj Close'].values, 12, 26, 9)
    data[ticker+' rsi'] = indicators.rsi(data[ticker+' Adj Close'].values, 14) #calculates RSI using 14 day window
    data['target'] = indicators.pct_diff(data[ticker+' Adj Close'].values, 9) #pct diff between day and day+X
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

def tsne_decompose(x,y_actions,ticker):
    # performs TSNE decomposition on your data
    actions = ["hard sell", "soft sell", "hold", "soft buy", "hard buy"]
    colors = ['darkviolet', 'dodgerblue', 'limegreen', 'orange', 'crimson']

    fig, axs = plt.subplots(1,1)

    for i,act in enumerate(actions):
        action_mask = [x == act for x in y_actions]
        foo_train = x[action_mask]

        tsne_train = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
        tsne_train_results = tsne_train.fit_transform(foo_train)

        axs.scatter( tsne_train_results[:,0],
                    tsne_train_results[:,1],
                    color=colors[i],
                    marker='.',
                    alpha=0.75,
                    label=act)


    axs.set_xlabel("T-SNE var 1")
    axs.set_ylabel("T-SNE var 2")
    axs.legend(loc='best')
    fig.savefig(f"./plots/tsne_decomp_{ticker}.png")

    plt.show()


def regress(args):
    """builds a regressor using xgboost to predict the 5 day pct diff for a given ticker"""

    X, y = get_data(args.ticker)
        

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(X.values,
                                                        y.values,
                                                        test_size=args.split)

    if args.tsne:
        # apply actions to y_vals
        y_actions = y.apply(what_change)

        # TNSE decomposition:
        tsne_decompose(X.values, y_actions, args.ticker)
        return

    # create regression ready dataset
    dm = xgb.DMatrix(data=x_train, label=y_train)

    # these were determined by the bayesian optimization procedure
    params = {"objective":'reg:squaredlogerror', 
                "eval_metric":'rmse',
                'colsample_bytree': 0.4582463307002723,
                #'gamma': 9.85616591125086, 
                'learning_rate': 0.1205329684150237, 
                'max_depth': 5, 
                'min_child_weight': 4.837582653206183, 
                'num_boost_round': 48.94484560995088, 
                'subsample': 0.6690714794699937
                }

    if args.hpo:
        # bayesian optimization of hyperparameters
        from bayes_opt import BayesianOptimization

        def fcv(max_depth, gamma, min_child_weight, subsample, colsample_bytree, learning_rate, num_boost_round):
            params = {"objective":'reg:squaredlogerror', 
                "eval_metric":'rmse',
                'subsample': 0.5,
                'gamma': 0.2,
                'colsample_bytree': 0.8,
                'learning_rate': 0.3, 
                'max_depth': 8, 
                'min_child_weight': 0.9,
                'num_boost_round': 53,
                }
            cv_results=xgb.cv(dtrain=dm, params=params, nfold=10, num_boost_round=int(num_boost_round), early_stopping_rounds=10, metrics='rmse', as_pandas=True)
            # will try to MAXIMIZE this function, but we want to get close to 0
            # make result negative to maximize
            return -cv_results['test-rmse-mean'].min() 

        # Now, create a dictionary for the boundaries we should search within
        dict_cv = {
          'max_depth': (2, 12),
          'gamma': (0.001, 10.0),
          'min_child_weight': (0, 20),
          'subsample': (0.4, 1.0),
          'colsample_bytree': (0.4, 1.0),
          'learning_rate': (0.1, 1.0),
          'num_boost_round': (0,100),
          }

        XGB_BO = BayesianOptimization(fcv, dict_cv)
        XGB_BO.maximize(init_points=10, n_iter=20, acq='ei', xi=0.0) 

        print("the optimal parameters were determined to be:")
        print(XGB_BO.max['params'])
    
        params = XGB_BO.max['params']
        return

    # train the xgboost model
    xg_m = xgb.train(params=params, dtrain=dm)

    # predict the output for the test values
    dm_test = xgb.DMatrix(data=x_test)
    xg_predict = xg_m.predict(dm_test)
    xg_m.dump_model(f"models/model_{args.ticker}.txt")

    # check how well we did
    mask = y_test != 0
    y_test = y_test[mask]
    xg_predict = xg_predict[mask]
    explained_variance = 1 - np.divide(np.var(np.subtract(y_test, xg_predict)), y_test)
    rms_err = np.sqrt(mean_squared_error(xg_predict,y_test))
    chi_sqr = np.sum( np.divide(np.power(np.subtract(xg_predict, y_test),2), np.abs(y_test)))
    xg_actions = [what_change(x) for x in xg_predict]
    y_actions = [what_change(x) for x in y_test]
    fakes = sum(('buy' in xg_actions[i] and 'sell' in y_actions[i]) or ('sell' in xg_actions[i] and 'buy' in y_actions[i]) for i in range(len(y_actions)))
    print(f'there are {len(y_test)} events in the test sample')
    print(f'the mean squared error is: {round(rms_err, 4)}')
    print(f'the reduced chi squared value is: {chi_sqr/(len(y_test)-1)}')
    print(f'the number of fakes is: {100*fakes/len(y_test)}')


    if args.ratio:
        # plot the information
        actions = ["hard sell", "soft sell", "hold", "soft buy", "hard buy"]
        colors = ['darkviolet', 'dodgerblue', 'limegreen', 'orange', 'crimson']
        fig, axs = plt.subplots(nrows=2, ncols=len(actions))

        for i,act in enumerate(actions):
            action_mask = [x == act for x in [what_change(z) for z in y_test]]
            y_vals = y_test[action_mask]
            x_vals = xg_predict[action_mask]
            y_actions = [what_change(x) for x in y_vals]
            x_actions = [what_change(x) for x in x_vals]
            explained_variance = [actions.index(y_actions[j]) - actions.index(x_actions[j]) for j in range(len(y_actions))]
            axs[0][i].plot(y_vals, label="true values", marker='.', linewidth=0)
            axs[0][i].plot(x_vals, label="predicted values", marker='.', linewidth=0)
            axs[0][i].legend(loc='best')
            axs[0][i].set_ylabel(f'x day pct change for {args.ticker}')
            axs[0][i].set_title(f"Results for {act}")
            axs[1][i].plot(explained_variance, label="True Action - Predicted Action", marker='.', linewidth=0)
            axs[1][i].set_ylabel("True Action - Predicted Action")
    
        plt.show()

