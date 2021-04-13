#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 08:49:59 2021

@author: gilto
"""

import numpy as np
import numba as nb

njit = nb.njit

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