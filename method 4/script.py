import numpy as np
import os
import sys
import glob
sys.path.append('../src/')
from functions import load_data, getListOfFiles



noise_path = '../data/noise_signals/'
snr = 'I_30db'
cycles_lists = np.array((os.listdir(noise_path)))
full_path = f'{noise_path}{cycles_lists[0]}/{snr}/'
samples = getListOfFiles(full_path)
# data, fault = load_data(full_path)
print(len(samples))
print(samples)
# print(cycles_lists[0])
# print(fault)
# print(data)