from funcs import open_data, plot_data
import matplotlib.pyplot as plt

data_path = 'data/'
model_path = 'models/'
fig_path = 'fig/'


data = open_data('noisy_data.pkl')
print(data[0].keys())