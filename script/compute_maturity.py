#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:55:00 2021

@author: gilto
"""

import numpy as np
import numba as nb

njit = nb.njit

# This function computes the Monte Carlo samples of final wealth and maturity
# for a given leval of dimensionless risk aversion level xi and a coupon value
# c. This function is callibrated to the desired maturity with a fixed xi.
@njit(nogil=True)
def compute_maturity(c, xi,  W0, r, m, sigma, K, S):
    np.random.seed(0)
    K_values = - np.ones(S)
    W_K_values = - np.ones(S)
    for s in range(S):
        W_k = W0
        k = 0
        c_discounted = c
        explodes = False
        while W_k >= c_discounted:
            mu_k = m + sigma*np.random.normal()
            coef = xi*(mu_k - r)/(1 + r)
            W_k = (1 + coef)*(W_k - c_discounted)
            if (W_k > 1e7):
                explodes = True
                break
            k += 1
            c_discounted /= 1 + r
        if not explodes:
            K_values[s] = k
            W_K_values[s] = W_k
    return K_values[K_values >= 0], W_K_values[W_K_values >= 0]