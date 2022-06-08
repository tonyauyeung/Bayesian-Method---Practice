"""
-*- coding = utf-8 -*-
@time:2022-06-01 20:51
@Author:Tony.SeoiHong.AuYeung
@File:MCMC.py
@Software:PyCharm
"""
import numpy as np
import tqdm
from utils import likelihood
import copy
from scipy.stats import multivariate_normal


def MH(data, n, sampler, T=1000):
#     sampler = lambda x: np.random.uniform(0, 2, size=x) if sampler is None else sampler
    samples = []
    sig2 = 5
    theta = sampler(n)
    samples.append(theta)
    for t in tqdm.tqdm_notebook(range(T - 1)):
        for i in range(n):
            theta_new = copy.deepcopy(theta)
            s = sampler(1)
#             theta_new[i] = sampler(1)
            theta_new[i] = s
            tmp1 = likelihood(data, theta_new)
            tmp2 = likelihood(data, theta)
            alpha = min(1, tmp1 / tmp2)
            u = np.random.uniform(0, 1, 1)
            if u <= alpha:
                theta = theta_new
        samples.append(theta)
    return np.stack(samples)


def RJMCMC(data, T=1000):
    model_prior = [0.1, 0.9]
    m = np.random.choice([1, 2], 1, p=model_prior)
    theta = np.random.normal(0, 2, 48)
    if m == 1:
        theta[24:] = np.zeros(24)
    samples = [theta]
    var = multivariate_normal(mean=np.zeros(24), cov=np.eye(24) * 2)
    for t in tqdm.tqdm_notebook(range(T - 1)):
        if m == 1:
            beta_new = np.random.normal(0, 2, 24)
            theta_new = copy.deepcopy(theta)
            theta_new[24:] = beta_new
            tmp1 = likelihood(data, theta_new)
            tmp2 = likelihood(data, theta)
            alpha = min(1, tmp1 * 0.9 / (tmp2 * 0.1))
            m += 1
        else:
            theta_new = copy.deepcopy(theta)
            deleted = theta_new[24:]
            theta_new[24:] = np.zeros(24)
            tmp1 = likelihood(data, theta_new)
            tmp2 = likelihood(data, theta)
            alpha = min(1, tmp1 * 0.1 / (tmp2 * 0.9))
            m -= 1
        u = np.random.uniform(0, 1, 1)
        if u <= alpha:
            theta = theta_new

        for i in range(24):
            theta_new = copy.deepcopy(theta)
            s = np.random.normal(0, 2, 1)
            theta_new[i] = s
            tmp1 = likelihood(data, theta_new)
            tmp2 = likelihood(data, theta)
            alpha = min(1, tmp1 / tmp2)
            u = np.random.uniform(0, 1, 1)
            if u <= alpha:
                theta = theta_new
        samples.append(theta)
    return np.stack(samples)