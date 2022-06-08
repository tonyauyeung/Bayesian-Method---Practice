"""
-*- coding = utf-8 -*-
@time:2022-06-01 20:42
@Author:Tony.SeoiHong.AuYeung
@File:main.py
@Software:PyCharm
"""
from utils import dataloader, prob
from plots import convergence_plot, acf_plot, trace_plot
import matplotlib.pyplot as plt
from MCMC import MH
import os
import pickle
from itertools import permutations
import numpy as np


N = 24


def inference(data, samples):
    index = data[67]
    l = int(len(index) / 2)
    T = samples.shape[0]
    f = []
    print(samples.shape)
    for sample in samples:
        if samples.shape[1] == 24:
            tmp = sample[index]
        else:
            tmp = sample[index][:l] + sample[index][l:]
        f.append(np.exp(tmp[5]) / np.sum(np.exp(tmp)))
    print('Prob of 7-th player win in the 68-th game: {}'.format(np.mean(f)))
    return f
    
    
def pipeline(data, trail_time, burn_in, file_path, MH, sampler, is_scratch=True, mode=2):
#     file_path += '_model-{}'.format(mode)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    os.chdir(file_path)
    if is_scratch:
        trails = []
        trail_time = trail_time
        burn_in = burn_in
        for i in range(3):
            trails.append(MH(data, N * mode, sampler, T=trail_time))
        with open('param.pkl', 'wb') as f:
            dic = {'T': trail_time, 'burn_in': burn_in, 'trails': trails}
            pickle.dump(file=f, obj=dic)
    else:
        with open('param.pkl', 'rb') as f:
            dic = pickle.load(f)
            trail_time, burn_in, trails = dic['T'], dic['burn_in'], dic['trails']
    samples = trails[-1]
    convergence_plot(trails, trail_time, burn_in)
    acf_plot(samples, lags=150)
    trace_plot(samples)
#     plt.show()
#     inference(data, samples)
#     if mode == 1:
#         inference(data_o, samples)
#     else:
#         inference(data_oe, samples)
    os.chdir('..')
    os.chdir('..')
    
    return samples

# trail_time = 20
# burn_in = 5
# pipeline(trail_time, burn_in, '/test', MH_normal, mode=1)
if __name__ == '__main__':
    data_o, data_oe = dataloader()
    trail_time = 20
    burn_in = 5
    sampler1 = lambda x: np.random.normal(0, 2, size=x)
    sampler2 = lambda x: np.random.uniform(0, 2, size=x)
    for i, data in enumerate([data_o, data_oe]):
        MH1 = MH(data, 24 * (i + 1), sampler1, T=trail_time)
        MH2 = MH(data, 24 * (i + 1), sampler2, T=trail_time)
        pipeline(trail_time, burn_in, 'normal/model{}'.format(i + 1), MH1, is_scratch=True, mode=i + 1)
        pipeline(trail_time, burn_in, 'uniform/model{}'.format(i + 1), MH2, is_scratch=True, mode=i + 1)