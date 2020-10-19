import sys
sys.path.append('../src/')
from functions import load_data, get_signal
import matplotlib.pyplot as plt


snr = 'I_pu'
data = list(get_signal('cycle_1', snr))
print(len(data))
print(data)
plt.plot(data[0]['signal'])
plt.show()