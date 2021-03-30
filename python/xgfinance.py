#!/usr/bin/env python3
"""
Main control script

This is the main control script for this xgboost-finance package.
"""

import click
import pandas as pd
import numpy as np

import scraper
import plotter
import booster

@click.command()
@click.option('-t','--ticker', type=str, default='', required=False)
@click.option('--extract', default=False, is_flag=True, required=False)
@click.option('--candle', default=False, is_flag=True, required=False)
@click.option('--corr', default=False, is_flag=True, required=False)
@click.option('--classify', default=False, is_flag=True, required=False)
@click.option('--split', default="traditional", required=False)
@click.option('--regress', default=False, is_flag=True, required=False)
def xgfinance(ticker, extract, candle, corr, classify, regress, split):
    """Run the xgboost finance package."""    

    #run the data extraction scripts
    if extract:
        #try:
        scraper.get_data()
        #except:
        #    print("[ERROR] something went wrong with the extraction")

    #check for plotting
    if candle:
        # try:
        plotter.plot_ticker(ticker)
        #except:
         #   print("[ERROR] plotter.plot_ticker({}) failed to run".format(ticker))
    
    if corr:
        plotter.plot_corr()

    if classify:
        booster.classify(ticker, split)
    if regress:
        booster.regress(ticker, split)

    #send everything over to the model builder and train

    #run the validation

if __name__ == '__main__':
    xgfinance()
