#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 08:49:59 2021

@author: gilto
"""

import numpy as np
import numba as nb

import matplotlib.pyplot as plt

njit = nb.njit

PNG_PATH = "../note/png/"
AWESOME_PURPLE = 0.55, 0.42, 1

# Computing the riskless rate discounted coupons
# tilde_c[k] = min(c/(1 + r)**k, tilde_W[k])
# The portfolio cannot deliver a coupon higher than what is available.
@njit(nogil=True)
def compute_c_discounted(W_k, c, r, k):
    c_discounted = c/np.power(1 + r, k)
    return min(c_discounted, W_k)

# Computing the wealth at every instants between 0 and K for all Monte-Carlo
# samples. As a reminder from the notes, the dynamics is computed in 
# dimensionless form using
# tilde_W[k + 1] = (1 + xi*(mu[k] - r)/(1 + r))*(tilde_W[k] - tilde_c[k])
@njit(nogil=True)
def compute_W(mu, W0, xi, c, r, S, K):
    W = np.empty((S, K + 1))
    for s in range(S):
        W[s, 0] = W0
        for k in range(K):
            W_k = W[s, k]
            c_discounted = compute_c_discounted(W_k, c, r, k)
            coef = xi*(mu[s, k] - r)/(1 + r)
            W[s, k + 1] = (1 + coef)*(W_k - c_discounted)
    return W

# Computing one wealth trajectory assuming no risk aversion - for illustrative
# purposes.
@njit(nogil = True)
def compute_illustrative_daily_wealth(W0, m, sigma, K, c):
    np.random.seed(0)
    W = np.empty(30*K + 1)
    W[0] = W0
    for k in range(30*K):
        mu_k = m/30 + sigma/np.sqrt(30)*np.random.normal()
        if k % 30 == 0:
            W[k] -= min(W[k], c)
        W[k + 1] = max(0, (1 + mu_k)*W[k])
    W[-1] -= min(W[-1], c)
    return W

# Plotting three trajectories to illustrate the importance of choosing the 
# right coupon for the investment.
def plot_wealth_examples(W0, m, sigma, K):
    parameter_key = f"W0={W0}_m={m}_sigma={sigma}_K={K}"
    plt.plot(compute_illustrative_daily_wealth(W0, m, sigma, K, c = 0.03), color = AWESOME_PURPLE, linewidth = .5)
    plt.plot(compute_illustrative_daily_wealth(W0, m, sigma, K, c = 0.048), color = AWESOME_PURPLE, linewidth = .5)
    plt.plot(compute_illustrative_daily_wealth(W0, m, sigma, K, c = 0.06), color = AWESOME_PURPLE, linewidth = .5)
    plt.xticks([0, 30*K + 1], ["0", r"Death"])
    plt.yticks([0, 1], [r"$0$", r"$W_0$"])
    plt.xlabel("Time")
    plt.ylabel("Wealth")
    plt.title("Wealth across time for different pension coupons")
    plt.savefig(PNG_PATH + f"wealth_examples_{parameter_key}.png")
