#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 08:10:29 2021

@author: gilto
"""

import numpy as np
import numba as nb

from compute_parameters import compute_m_and_sigma
from fixed_maturity import run_for_fixed_maturity
from variable_maturity import run_for_variable_maturity

njit = nb.njit
np.random.seed(0)

PNG_PATH = "../note/png/"
AWESOME_PURPLE = 0.55, 0.42, 1

W0 = 1                 # For easier computations - results are linear anyway in W0
r = 0.005              # taux livret A
K = 24                 # Consider a whole year of investment
S = 100000             # Number of Monte-Carlo simulations

# Extracting average and standard deviation of Lyxor ETF CAC 40
m, sigma = compute_m_and_sigma(resample_frequency = "1M")

# First runs the analysis for a fixed maturity. All plots go 
# into the "../note/png" folder.
run_for_fixed_maturity(W0, r, m, sigma, K, S)

# Second runs the analysis considering a variable maturity. All plots go 
# into the "../note/png" folder.
run_for_variable_maturity(W0, r, m, sigma, K, S)