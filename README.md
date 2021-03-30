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
    - Target (buy/sell/hold : 1/-1/0) will be defined based on a 3% difference over a 5 day period
4) split and shape the data for train and test using TimeSeriesSplit
    - Tried splitting as Time Series with lackluster result
    - Tried splitting using train_test_split with improved performace
5) train our model in xgboost
    - Model (classifier) seems to perform with mixed results when using Time Series split and buy/sell/hold target. Accuracy ranges from ~20% to ~75%.
    - Model (classifier) seems to perform better when using a tranditional 80:20 split with our buy/sell/hold target. Accuracy sits around ~83%.
    - Model (regressor) was built to target the X day % diff. Performance is varied, but overall the method seems to have a RMS Error of about 1 - 2. This RMS Error improves as X increases.
6) validate the model using shap (https://github.com/slundberg/shap)

## Usage
1) obtain the data  
`./python/xgfinance.py --extract`
2) plot the data   
`./python/xgfinance.py -t [TICKER] --candle`
`./python/xgfinance.py --corr`  
3) build a model  
`./python/xgfinance.py -t [TICKER] --classify --split=[SPLIT_METHOD]`
 
## To Investigate
How to properly account for time dependence of data?
   - Should we split using Time Series?
   - Are there options for XGBoost to look at past data?
   - Should additional columns with past info be included? This strikes me as being a very plausible option
Should additional metrics be included?
    - This will likely be answered by either testing them directly or by checking shap outputs, or both. 

