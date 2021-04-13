#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:56:39 2021

@author: gilto
"""

import numpy as np
import numba as nb

import scipy.optimize as optimize

import matplotlib.pyplot as plt

from compute_maturity import compute_maturity

njit = nb.njit

PNG_PATH = "../note/png/"
AWESOME_PURPLE = 0.55, 0.42, 1

# This runs all the analysis with variable maturity. It saves in the 
# "../note/png/" folder the plots of the coupon curve, the gains (additional
# wealth), the expected time to zero wealth and finally the expected shortfall.
def run_for_variable_maturity(W0, r, m, sigma, K, S):
    
    rho_max = 1/(1 + r)*(1.64 - (m - r)/sigma)
    
    step = 0.01
    xi_values = np.arange(0, 1/rho_max + step, step)         # xi
    c_values = np.empty_like(xi_values)                      # coupon
    K_mean_values = np.empty_like(xi_values)                 # life span
    K_std_values = np.empty_like(xi_values)                  # life span std
    additional_wealth_mean_values = np.empty_like(xi_values) # additional wealth
    additional_wealth_std_values = np.empty_like(xi_values)  # additional wealth std
    VaR_5_values = np.empty_like(xi_values)                  # Value At Risk 5%
    ES_5_values = np.empty_like(xi_values)                   # expected shortfall
    STDS_5_values = np.empty_like(xi_values)                 # std shortfall
    
    for n, xi in enumerate(xi_values):
        
        def objective(c):
            K_values, _ = compute_maturity(c, xi,  W0, r, m, sigma, K, S)
            K_mean = K_values.mean()
            return K_mean - K
    
        print(f"Calculating coupon for xi = {xi}")
        c_value = optimize.root_scalar(objective, bracket = [W0/K, 1], method = 'brentq').root
    
        K_values, W_K_values = compute_maturity(c_value, xi,  W0, r, m, sigma, K, S)
    
        dW = ((1 + 1/r)*(1 - 1/np.power(1 + r, K_values))*c_value - W0) + W_K_values
        dW.sort()
        VaR_5 = dW[int(.05*dW.shape[0])]
    
        c_values[n] = c_value
        K_mean_values[n] = K_values.mean()
        K_std_values[n] = K_values.std()
        additional_wealth_mean_values[n] = dW.mean()
        additional_wealth_std_values[n] = dW.std()
        VaR_5_values[n] = VaR_5
        ES_5_values[n] = - dW[dW <= VaR_5].mean()
        STDS_5_values[n] = dW[dW <= VaR_5].std()    
    
    parameter_key = f"_non_fixed_W0={W0}_r={r}_m={m}_sigma={sigma}_K={K}_S={S}"
    
    # Plotting coupon curve with linear benchmark
    # Because of numerical inaccuracies, benchmark is set to first value found by 
    # optimizer
    benchmark = c_values[0]
    
    plt.figure()
    plt.plot(xi_values, c_values, color = AWESOME_PURPLE)
    plt.plot(xi_values, benchmark*np.ones_like(xi_values), ":", color = AWESOME_PURPLE)
    plt.title(r"Coupon value as function of $\xi$")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"Coupon value")
    plt.savefig(PNG_PATH + f"coupon_curve_{parameter_key}.png")
    
    # Plotting average life span
    plt.figure()
    plt.plot(xi_values, K_mean_values, color = AWESOME_PURPLE)
    plt.plot(xi_values, K_mean_values - K_std_values, "--", color = AWESOME_PURPLE)
    plt.plot(xi_values, K_mean_values + K_std_values, "--", color = AWESOME_PURPLE)
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
