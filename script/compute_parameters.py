#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:57:05 2021

@author: gilto
"""

import pandas as pd
import numpy as np

DATA_PATH = "../data/"

# Computing average returns and standard deviations of Lyxor ETF CAC 40. 
# Available data spans from 1st April 2020 to 20 March 2021
# Source: https://www.abcbourse.com/download/valeur/CACp
def compute_m_and_sigma(resample_frequency = "3M"):
    
    # Read data from file
    etf = pd.read_csv(DATA_PATH + "CAC.txt", sep = ";", names = ["STOCK_ID", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
    
    # Format dates and pivot table on close
    etf.DATE = etf.DATE.str[:-2] + "20" + etf.DATE.str[-2:]
    etf.DATE = pd.to_datetime(etf.DATE, format = "%d/%m/%Y")
    etf = etf[["DATE", "STOCK_ID", "CLOSE"]]
    etf = etf.rename(columns = {"CLOSE": "PRICE"})
    etf = pd.pivot_table(etf, values='PRICE', index=['DATE'], columns=['STOCK_ID'])
    
    # Daily resampling with an interpolation and then resampling again at
    # desired frequency.
    etf = etf.resample("1D").last().interpolate().resample(resample_frequency).last()
    
    # Computing returns
    returns = etf.copy()
    for stock in returns.columns:
        returns[stock] = np.log(returns[stock]/returns[stock].shift(1)).values
    
    # Desired parameters
    m = returns.mean().values[0]
    sigma = returns.std().values[0]

    print(f"for resampling_frequency = {resample_frequency} m = {m} and sigma = {sigma}")

    return m, sigma