# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:07:35 2021

@author: gacoc
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt

def risky_returns(m, sigma, size):
    return m + sigma*np.random.normal(size = size)

def bond(c, r):
    return (c/np.power(1 + r, np.arange(c.shape[0]))).sum()

def wealth(w0, mu, r, psi, c):
    risky = np.dot((mu - r)/np.power(1 + r, 1 + np.arange(psi.shape[0])), psi)
    bond_ = bond(c, r)
    return w0 + risky - bond_

def all_wealths(w0, mu, r, psi, c, S):
    wealths = np.zeros((mu.shape[0], mu.shape[1]))
    for k in range(psi.shape[0]):
        wealths[:, k] = wealth(w0, mu[:, :(k + 1)], r, psi[:(k + 1)], c[:(k + 1)])
    return wealths

def proba_superior_riskless(w, w0):
    w0_ = w0*np.ones_like(w)
    for k in range(1, 1 + w.shape[1]):
        w0_[:, k - 1] *= (1 - k/(w0_.shape[1]))
    return w > w0_

def proba_no_debt(w, r, psi, c):
    return (w > (psi + c)/np.power(1 + r, np.arange(c.shape[1])))

def find_strategy(w0, r, m, sigma, K, alpha, S):
    bounds = opt.Bounds(0, np.inf)
    mu = risky_returns(m, sigma, (S, K))
    x0 = np.array([(1 + r)**k*w0/K for k in np.arange(2*K)])
    def f(x):
        c = x[K:]
        return - bond(c, r)
    constraints = list()
    # def constraint_superior_riskless(x):
    #     psi = x[:K]
    #     c = x[K:]
    #     wealths = all_wealths(w0, mu, r, psi, c, S)
    #     p = proba_superior_riskless(wealths, w0).prod(axis = 1).mean()
    #     print(f"constraint_superior_riskless = {p}")
    #     return np.atleast_1d(p + alpha - 1)
    # constraints.append({"fun": constraint_superior_riskless, "type": "eq"})   
    def constraint_no_debt(x):
        psi = x[:K]
        c = x[K:]
        print(f"psi = {psi}")
        print(f"c = {c}")
        wealths = all_wealths(w0, mu, r, psi, c, S)
        psi = np.vstack(S*(x[:K].tolist(), ))
        c = np.vstack(S*(x[K:].tolist(), ))
        p = proba_no_debt(wealths, r, psi, c).prod(axis = 1).mean()
        print(f"constraint_no_debt = {p}")
        return np.atleast_1d(p - 1)
    constraints.append({"fun": constraint_no_debt, "type": "eq"})   
    results = opt.minimize(f, x0, bounds = bounds, constraints = constraints)
    return results
    
if __name__ == "__main__":

    w0 = 15
    r = 0.01
    m = 0.04
    sigma = 1
    alpha = .05
    K = 3
    S = 10
    print(find_strategy(w0, r, m, sigma, K, alpha, S))
    
    # psi = np.random.uniform(low = .5*w0/K, high = w0/K, size = K)
    # c = np.random.uniform(low = .75*w0/K, high = w0/K, size = K)
    # print(f"psi = {psi}")
    # print(f"c = {c}")

    # wealths = all_wealths(w0, m, r, psi, c, S)
    # for wealth_ in wealths:
    #     print("***")
    #     print(f"Wk = {wealth_}")
    #     print(f"proba_superior_riskless = {proba_superior_riskless(wealth_, w0)}")
    #     print(f"proba_no_debt = {proba_no_debt(wealth_, r, psi, c)}")