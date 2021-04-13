#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:09:36 2021

@author: gilto
"""

import numpy as np
import numba as nb

import matplotlib.pyplot as plt

from compute_wealth_trajectories import compute_W, compute_c_discounted
from compute_coupon import get_coupon

njit = nb.njit

PNG_PATH = "../note/png/"
AWESOME_PURPLE = 0.55, 0.42, 1

# Helper function to avoid repetitive typing
def get_stats(x):
    x_std = x.std()
    return x.mean(), x_std

# The additional wealth is computed as the difference between the sum of all
# discounted coupons and the initial wealth. As the investor is less risk 
# averse, the additional wealth distribution widens with a positive average
# if average returns of risky asset are positive
@njit(nogil=True)
def compute_additional_wealth(W0, W, c_value, r, S, K):
    additional_wealth = np.empty(S)
    for s in range(S):
        additional_wealth[s] = 0
        for k in range(K + 1):
            additional_wealth[s] += compute_c_discounted(W[s, k], c_value, r, k)
    return additional_wealth - W0

# This runs all the analysis for a fixed maturity. It saves in the 
# "../note/png/" folder the plots of the coupon curve, the gains (additional
# wealth), the expected time to zero wealth and finally the expected shortfall.
def run_for_fixed_maturity(W0, r, m, sigma, K, S):
    
    parameter_key = f"W0={W0}_r={r}_m={m}_sigma={sigma}_K={K}_S={S}"
    
    print(f"Running fixd maturity analysis for {' '.join(parameter_key.split('_'))}")
    
    rho_max = 1/(1 + r)*(1.64 - (m - r)/sigma)

    mu = m + sigma*np.random.normal(size = (S, K))
    step = 0.01
    xi_values = np.arange(0, 1/rho_max + step, step)         # xi
    c_values = np.empty_like(xi_values)                      # coupon
    T_mean_values = np.empty_like(xi_values)                 # life span
    T_std_values = np.empty_like(xi_values)                  # life span std
    additional_wealth_mean_values = np.empty_like(xi_values) # additional wealth
    additional_wealth_std_values = np.empty_like(xi_values)  # additional wealth std
    VaR_5_values = np.empty_like(xi_values)                  # Value At Risk 5%
    ES_5_values = np.empty_like(xi_values)                   # expected shortfall
    STDS_5_values = np.empty_like(xi_values)                 # std shortfall
    
    for n, xi in enumerate(xi_values):
        
        print(f"Calculating coupon for xi = {xi}")
        c_value = get_coupon(mu, W0, xi, r, S, K)[0]
        
        # Computing risk management statistics from Monte-Carlo samples
        W = compute_W(mu, W0, xi, c_value, r, S, K)
        
        # stopping time stats (average life span and std)
        stop = get_stats((W == 0).sum(axis = 1))
        
        # VaR computation
        dW = compute_additional_wealth(W0, W, c_value, r, S, K)
        dW.sort()
        additional_wealth = get_stats(dW)
        VaR_5 = dW[int(.05*dW.shape[0])]
        
        # Storing values
        c_values[n] = c_value
        T_mean_values[n] = K - stop[0] + 1
        T_std_values[n] = stop[1]
        additional_wealth_mean_values[n] = additional_wealth[0]
        additional_wealth_std_values[n] = additional_wealth[1]
        VaR_5_values[n] = VaR_5
        ES_5_values[n] = - dW[dW <= VaR_5].mean()
        STDS_5_values[n] = dW[dW <= VaR_5].std()    
    
    # Plotting coupon curve with linear benchmark
    benchmark = r/(1 + r)*np.power(1 + r, K)/(np.power(1 + r, K) - 1)*W0
    
    plt.figure()
    plt.plot(xi_values, c_values, color = AWESOME_PURPLE)
    plt.plot(xi_values, benchmark*np.ones_like(xi_values), ":", color = AWESOME_PURPLE)
    plt.title(r"Coupon value as function of $\xi$")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"Coupon value")
    plt.savefig(PNG_PATH + f"coupon_curve_{parameter_key}.png")
    
    # Plotting average life span
    plt.figure()
    plt.plot(xi_values[1:], T_mean_values[1:], color = AWESOME_PURPLE)
    plt.plot(xi_values[1:], T_mean_values[1:] - T_std_values[1:], "--", color = AWESOME_PURPLE)
    plt.plot(xi_values[1:], T_mean_values[1:] + T_std_values[1:], "--", color = AWESOME_PURPLE)
    plt.title(r"Average life span of investment as function of $\xi$")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"Average life span of investment")
    plt.savefig(PNG_PATH + f"average_life_span_{parameter_key}.png")
    
    # Plotting additional wealth evolution
    plt.figure()
    plt.plot(xi_values, additional_wealth_mean_values, color = AWESOME_PURPLE)
    plt.plot(xi_values, additional_wealth_mean_values - additional_wealth_std_values, "--", color = AWESOME_PURPLE)
    plt.plot(xi_values, additional_wealth_mean_values + additional_wealth_std_values, "--", color = AWESOME_PURPLE)
    plt.title(r"Average additional wealth as function of $\xi$")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"Additional wealth")
    plt.savefig(PNG_PATH + f"additional_wealth_{parameter_key}.png")
    
    # Plotting expected shortfall evolution
    plt.figure()
    plt.plot(xi_values, ES_5_values, color = AWESOME_PURPLE)
    plt.plot(xi_values, ES_5_values - STDS_5_values, "--", color = AWESOME_PURPLE)
    plt.plot(xi_values, ES_5_values + STDS_5_values, "--", color = AWESOME_PURPLE)
    plt.title(r"5% Expected shortfall as function of $\xi$")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"5% Expected shortfall")
    plt.savefig(PNG_PATH + f"expected_shortfall_{parameter_key}.png")