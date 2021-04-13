# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 18:53:27 2021

@author: gacoc
"""

import pandas as pd
import numpy as np

etf = pd.read_csv("CAC.txt", sep = ";", names = ["STOCK_ID", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
etf.DATE = etf.DATE.str[:-2] + "20" + etf.DATE.str[-2:]
etf.DATE = pd.to_datetime(etf.DATE, format = "%d/%m/%Y")
etf = etf[["DATE", "STOCK_ID", "CLOSE"]]
etf = etf.rename(columns = {"CLOSE": "PRICE"})
etf = pd.pivot_table(etf, values='PRICE', index=['DATE'], columns=['STOCK_ID'])
etf = etf.resample("1D").last().interpolate().resample("1M").last()
returns = etf.copy()
for stock in returns.columns:
    returns[stock] = np.log(returns[stock]/returns[stock].shift(1)).values

m = returns.mean().values[0]
sigma = returns.std().values[0]

print(f"From April 1st, 2020 to March 5th, 2021, CAC40 etf stats are m = {m:.4f}\tsigma = {sigma:.4f}")

