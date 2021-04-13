#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 08:10:29 2021

@author: gilto
"""

from compute_wealth_trajectories import plot_wealth_examples
from compute_parameters import compute_m_and_sigma
from fixed_maturity import run_for_fixed_maturity
from variable_maturity import run_for_variable_maturity

PNG_PATH = "../note/png/"
AWESOME_PURPLE = 0.55, 0.42, 1

W0 = 1                 # For easier computations - results are linear anyway in W0
r = 0.005/12           # livret A rate - monthalized
K = 24                 # Consider a whole year of investment
S = 100000             # Number of Monte-Carlo simulations

# Extracting average and standard deviation of Lyxor ETF CAC 40
m, sigma = compute_m_and_sigma(resample_frequency = "1M")

# Plotting illustrative trajectories of wealth with regularly paid coupons
plot_wealth_examples(W0, m, sigma, K)

# First runs the analysis for a fixed maturity. All plots go 
# into the "../note/png" folder.
run_for_fixed_maturity(W0, r, m, sigma, K, S)

# Second runs the analysis considering a variable maturity. All plots go 
# into the "../note/png" folder.
run_for_variable_maturity(W0, r, m, sigma, K, S)