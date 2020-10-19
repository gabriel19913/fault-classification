import sys
sys.path.append('../src/')
import tsfel
import pandas as pd
from functions import get_signal, getListOfFiles, load_data

# # load dataset
# df = pd.read_csv('Dataset.txt')
noise_path = '../data/noise_signals/'
cycle = 'cycle_1'
noise = 'I_pu'
full_path = f'{noise_path}{cycle}/{noise}/'
samples = getListOfFiles(full_path)
data, fault = load_data(samples[0])

# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain()

# Extract features
X = tsfel.time_series_features_extractor(cfg, data)
print(X)