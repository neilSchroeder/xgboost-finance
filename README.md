# xgboost-finance
A second attempt at financial analysis, this time using xgboost

## Motivation
This is just a personal project I find interesting. I've learned that xgboost is an industry standard tool for model development so I'm going to try to build a fairly simply classifier for buying/selling/holding S&P 500 stocks  
Under no circumstances should anyone feel that this algorithm is successful without first understanding it themselves, nor is this to be considered financial advice in any way. I am not responsible for what you decide to use this code for.

## Outline
1) Obtain the data
    - We will scrape the wiki for s&p 500 to get the tickers
    - We'll then pull the data for all the tickers from yahoo
    - With data in hand we'll do a bit of visualization and maybe some correlation checks
2) Clean the data
3) Build our target
    - Build MACD (12, 26, 9)
    - Build RSE (14)
    - Build X day % diff
    - Target (hard buy/soft buy/hold/soft sell/hard sell : -5, -1, 0, 1, 5) will be defined based on a X% difference over a N day period
4) split and shape the data for train and test using TimeSeriesSplit
    - Tried splitting as Time Series with lackluster result
    - Tried splitting using train_test_split with improved performace
5) break down the model using T-SNE Decomposition to see if the models can be linear
    - The model doesn't look even remotely linear (see plots/tsne_decomp_MMM.png)
5) train our model in xgboost
    - Get an initial estimate of the model from the model hyperparameters
        - later we do hyperparameter tuning using bayesian optimization using https://github.com/fmfn/BayesianOptimization
    - train the model
        - I've chosen `reg:squaredlogerror` as the loss function since the target is compressed from 0 to 1. 
    - predict the outcomes
    - evaluate performance:
        - Check the fake rates (sell when you should buy or buy when you should sell)
        - mean squared error
        - reduced chi squared (this isn't a great metric, but it'll do for now)
    - plot the different categories and the category label differences

## Usage
1) obtain the data  
`./python/xgfinance.py --extract`
2) plot the data   
`./python/xgfinance.py -t [TICKER] --candle`
`./python/xgfinance.py --corr`  
3) build a model  
`./python/xgfinance.py -t [TICKER] --classify --split=[SPLIT_METHOD]`
 
## To Investigate
1) How can we outperform the "time-in" method?

