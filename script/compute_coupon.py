#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:25:40 2021

@author: gilto
"""

import numba as nb

from compute_wealth_trajectories import compute_W

njit = nb.njit

EPSILON = 1e-16

# Finding the first root of the average condition. As a reminder, the coupon
# value is optimal when
# < tilde_W[K] > = 0
# Say this optimal is c0, for all values above c0 are zero. Therefore the root
# is found with a modified bisection algorithm.
# The algorithm keeps a non-zero lower bracket value and a zero upper bracket
# value

@njit(nogil=True)
def get_coupon(mu, W0, xi, r, S, K):
    
    def objective(c):
        W = compute_W(mu, W0, xi, c, r, S, K)
        return W[:, -1].mean()
    
    c_start = 0
    c_end = 1
    objective_start = objective(c_start)
    objective_end = objective(c_end)
    old_c_end = None
    old_objective_end = None
    
    while objective_start > EPSILON:
        
        if objective_end == 0:
            
            old_c_end = c_end
            old_objective_end = objective_end
            c_end = (c_start + c_end)/2
            objective_end = objective(c_end)
        
        else:
            
            c_start = c_end
            c_end = old_c_end
            objective_start = objective_end
            objective_end = old_objective_end
            
    return c_start, objective_start

