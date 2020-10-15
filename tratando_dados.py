# %%
#%load_ext autoreload
#%autoreload 2
# %%
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from funcs import apply_noise, get_cycles, save_data, open_data, convert_label
from cumulantes import apply_cum

# %%
sns.set_context("talk")

# data_path = 'data/'
model_path = 'models/'
fig_path = 'fig/'

# with open(data_path + 'data.pkl', 'rb') as f:
#     data = pickle.load(f)

data = open_data('data.pkl')

# %%
V_pu_data = [{'V_pu': d['V_pu'],
              'fault_type': d['faultType']}
             for d in data]
# %%
I_pu_data = []
for i, v in enumerate(data):
    I_pu = data[i]['I_det'] / 609
    I_pu_data.append({'I_pu': I_pu, 'fault_type': data[i]['faultType']})

# %%
v_data = apply_noise(V_pu_data, 'V_pu', 'V')
# %%
i_data = apply_noise(I_pu_data, 'I_pu', 'I')
# %%
v_data = get_cycles(v_data, 'V')
# %%
# %%
i_data = get_cycles(i_data, 'I')

# %%
convert_label(v_data)
convert_label(i_data)
# %%
v_data_cum = apply_cum(v_data)
# %%
i_data_cum = apply_cum(i_data)
# %%
# save_data('v_noise_data_cum.pkl', v_data_cum)
# save_data('i_noise_data_cum.pkl', i_data_cum)
