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
    - Build 5 day % diff
    - Target (buy/sell/hold : 1/-1/0) will be defined based on a 3% difference over a 5 day period
4) split and shape the data for train and test
5) train our model in xgboost
6) validate the model using shap (https://github.com/slundberg/shap)

## Usage
1) obtain the data  
`./python/xgfinance.py --extract`
2) plot the data   
`./python/xgfinance.py -t [TICKER] --candle`
`./python/xgfinance.py --corr`  
3) build a model
 

