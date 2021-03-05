#!/usr/bin/env python3
"""
Main control script

This is the main control script for this xgboost-finance package.
"""

import click
import pandas as pd
import numpy as np

import scraper

@click.command()
@click.option('-t','--ticker', type=str, default='', required=False)
@click.option('--extract', default=False, is_flag=True, required=False)
def xgfinance(ticker, extract):
    """Run the xgboost finance package."""    

    #do some option management (if necessary)

    #run the data extraction scripts
    if extract:
        #        try:
        scraper.get_data()
        #except:
        #    print("[ERROR] something went wrong with the extraction")

    #send everything over to the model builder and train

    #run the validation

if __name__ == '__main__':
    xgfinance()
#except:
#    print("[ERROR] something went wrong in main")
