"""
-*- coding = utf-8 -*-
@time:2022-06-01 20:42
@Author:Tony.SeoiHong.AuYeung
@File:utils.py
@Software:PyCharm
"""
import re
import numpy as np


def dataloader():
    raw_data = ''
    with open('CompRank22.txt', 'r') as f:
        for line in f:
            raw_data += line.strip()
    raw_data = raw_data[5:-1].split('list')[1:]
    data = []
    for d in raw_data:
        tmp = re.findall(r'\d+', d)
        ind = int(len(tmp) / 2)
        data.append(np.array([tmp[: ind], tmp[ind:]], dtype=int) - 1)

    data_o = [tmp[0] for tmp in data]
    data_oe = [np.concatenate([tmp[0], tmp[1] + 24]) for tmp in data]
    return data_o, data_oe


def prob(index, theta):
    tmp = np.exp(theta[index] - np.max(theta[index]))
    return np.prod(tmp / np.sum(tmp))


def likelihood(data, theta):
    l = 1
    for d in data:
        l *= prob(d, theta)
    return l