import sys
sys.path.append('../src/')
from functions import data_list, apply_noise, save_data


data_path = '../data/detected_signals/'
model_path = 'models/'
fig_path = 'figs/'

data_list = data_list(data_path)
noise_data = apply_noise(data_list, min=10, max=100, s='I')
for i in range(len(noise_data)):
    save_data(noise_data[i], s=i, source='current')
