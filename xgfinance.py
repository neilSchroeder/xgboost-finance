#!/usr/bin/env python3
"""
Main control script

This is the main control script for this xgboost-finance package.
"""

import argparse as ap
import pandas as pd
import numpy as np

import python.scraper as scraper
import python.plotter as plotter
import python.booster as booster

def main():
    """Run the xgboost finance package."""    

    # option handling
    parser = ap.ArgumentParser(description="option handler for xgfinance")
    parser.add_argument("-t","--ticker", type=str, default='MMM',
                        help="ticker label to analyze")
    parser.add_argument("--extract", default=False, action="store_true", 
                        help="run the extraction scripts")
    parser.add_argument("--candle", default=False, action="store_true",
                        help="plot candle plots")
    parser.add_argument("--corr", default=False, action="store_true",
                        help="plot correlations")
    parser.add_argument("--split", default=0.2, type=float,
                        help="fraction of data to be used to test the model")
    parser.add_argument("--tsne", default=False, action="store_true",
                        help="visualize the linearity of the data through T-SNE decomposition")
    parser.add_argument("--hpo", default=False, action="store_true",
                        help="Run hyper-parameter optization, warning: this can take a long time")
    parser.add_argument("--ratio", default=False, action="store_true", 
                        help="plot ratio plots for the 5 categories")
    parser.add_argument("--walk", default=False, action="store_true",
                        help="activates the walk-forward method of evaluating the model")
    args = parser.parse_args()

    # run the data extraction scripts
    if args.extract:
        scraper.get_data()

    # make a candle plot of the ticker
    if args.candle:
        plotter.plot_ticker(args.ticker)

    # make a correlation plot of all data in the S&P 500
    if args.corr:
        plotter.plot_corr()

    # start the regression
    booster.regress(args)


if __name__ == '__main__':
    main()