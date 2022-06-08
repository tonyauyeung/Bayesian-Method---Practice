"""
-*- coding = utf-8 -*-
@time:2022-06-01 20:42
@Author:Tony.SeoiHong.AuYeung
@File:plots.py
@Software:PyCharm
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics import tsaplots

N = 24


def convergence_plot(trails, trail_time, burn_in):
    plt.subplots(figsize=(16, 8))
    plt.subplot(121)
    index = np.arange(burn_in, trail_time) + 1
    plt.plot(index, trails[0][burn_in:, 0], 'r')
    plt.plot(index, trails[1][burn_in:, 0], 'g')
    plt.plot(index, trails[2][burn_in:, 0], 'b')

    ax = plt.subplot(122)
    sns.kdeplot(trails[0][burn_in:, 0], ax=ax, color='r')
    sns.kdeplot(trails[1][burn_in:, 0], ax=ax, color='g')
    sns.kdeplot(trails[2][burn_in:, 0], ax=ax, color='b')

    plt.savefig('convergence_plot.png')


#     plt.show()

def trace_plot(samples):
    T = len(samples)
    x = np.arange(T) + 1
    plt.figure()
    plt.subplots(figsize=(32, 16))
    for i in range(24):
        plt.subplot(12, 2, i + 1)
        plt.plot(x, samples[:, i])
    plt.savefig('lambda_trace.png')

    if len(samples[0]) == 2 * N:
        plt.figure()
        plt.subplots(figsize=(32, 16))
        for i in range(24):
            plt.subplot(12, 2, i + 1)
            plt.plot(x, samples[:, i + N])
        plt.savefig('beta_trace.png')


def acf_plot(samples, lags=150):
    T = len(samples)
    plt.figure()
    plt.subplots(figsize=(16, 16))
    for i in range(24):
        ax1 = plt.subplot(4, 6, i + 1)
        tsaplots.plot_acf(samples[:, i], lags=lags, ax=ax1)
        plt.title('')
    plt.savefig('lambda_acf.png')

    if len(samples[0]) == 2 * N:
        plt.figure()
        plt.subplots(figsize=(16, 16))
        for i in range(24):
            ax2 = plt.subplot(4, 6, i + 1)
            tsaplots.plot_acf(samples[:, i + N], lags=lags, ax=ax2)
            plt.title('')
        plt.savefig('beta_acf.png')