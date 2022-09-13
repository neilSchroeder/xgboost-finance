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
4) Build a 3 week (21 day) sliding window 
    - XGBoost does not really do transformers like LSTM or RNN, so we need to have some input about the past
    - The way we do this is by creating new labels which are past information of each label
    - This significantly improves our forecasting ability, but isn't as powerful as a transformer
5) Split the dataset using `SKLearn train_test_split`.
    - Splitting time series data is an odd beast to tackle, but with the sliding window we can sort of avoid issues with this approach.
6) Break down the model using T-SNE Decomposition to see if the model can be described nicely
    - For highly dimensional datasets it's more useful to break down the dataset into ~50 components using PCA decomposition
    - After doing PCA decomposition, perform T-SNE decomposition on the PCA variables.
    - The model doesn't look even remotely linear (see plots/tsne_decomp_MMM.png)
7) Train our model in xgboost
    - Get an initial estimate of the model from the model hyperparameters
        - later we do hyperparameter tuning using bayesian optimization using https://github.com/fmfn/BayesianOptimization
    - train the model
        - I've chosen `reg:squaredlogerror` as the loss function since the target is compressed from 0 to 1. 
    - predict the outcomes
    - evaluate performance:
        - Check the fake rates (sell when you should buy or buy when you should sell)
        - mean squared error
    - plot the different categories and the category label differences (fake rate)
8) Bayesian hyperparameter tuning using `BayesianOptimization`
    - There are a couple different metrics you can aim for in this method
        - Area Under Curve: best performance was about 60%
        - Mean Squared Error: name says it all
        - Log Loss: an approach using entropy
    - XGBoost uses a number of parameters that are useful to optimize:
        - 

## Usage
1) obtain the data  
`./python/xgfinance.py --extract`
2) plot the data   
`./python/xgfinance.py -t [TICKER] --candle`
`./python/xgfinance.py --corr`  
3) build a model and check the fake rate
`./python/xgfinance.py -t [TICKER] --ratio`
 

