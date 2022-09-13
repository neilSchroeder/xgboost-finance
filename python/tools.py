import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

def what_change(pct_change, cut_sell, cut_buy):
    # determines what action to take based on the pct change    
    ret = []

    for x in pct_change:
        action = "hold"
        if x < cut_sell:
            action = "sell"
        if x > cut_buy:
            action = "buy"
        ret.append(action)

    return ret

def hyper_param_optimize(x, y, cut_sell, cut_buy):
    # bayesian optimization of hyperparameters

    def accuracy(preds, dtrain):
        return 'accuracy', accuracy_score(what_change(preds, cut_sell, cut_buy), what_change(dtrain.get_label(), cut_sell, cut_buy))


    dm = xgb.DMatrix(data=x, label=y)
    def fcv(max_depth, gamma, min_child_weight, subsample, colsample_bytree, learning_rate, num_boost_rounds):
        params = {"objective":'reg:squaredlogerror', 
            "eval_metric":'rmse',
            'subsample': subsample,
            'gamma': gamma,
            'colsample_bytree': colsample_bytree,
            'learning_rate': learning_rate, 
            'max_depth': int(max_depth), 
            'min_child_weight': min_child_weight,
            }
        cv_results=xgb.cv(dtrain=dm, 
                        params=params,
                        num_boost_round=int(num_boost_rounds),
                        nfold=10, 
                        early_stopping_rounds=10, 
                        custom_metric=accuracy,
                        as_pandas=True)
        return cv_results['test-accuracy-mean'].max()
    # Now, create a dictionary for the boundaries we should search within
    dict_cv = {
      'max_depth': (2, 12),
      'gamma': (0, 0.05),
      'min_child_weight': (0, 20),
      'subsample': (0.4, 1.0),
      'colsample_bytree': (0.4, 1.0),
      'learning_rate': (0.1, 1.0),
      'num_boost_rounds': (0,100)
      }
    XGB_BO = BayesianOptimization(f=fcv, pbounds=dict_cv)
    XGB_BO.maximize(init_points=25, n_iter=100) 
    print("the optimal parameters were determined to be:")
    print(XGB_BO.max['params'])

    params = XGB_BO.max['params']
    return params