from noise import decompress_pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA, PCA
INPUT_DATA_PATH = '../input-data/'

# data = decompress_pickle(INPUT_DATA_PATH + 'cycle_data')
# sample = data[0]
# X = sample['i_cycle_2'].reshape((4, 192)).T
# ica = FastICA(n_components=4)
# S_ = ica.fit_transform(X)  # Reconstruct signals
# A_ = ica.mixing_  # Get estimated mixing matrix

# f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
# ax[0][0].plot(X[:,0])
# ax[0][0].set_title('Phase A')
# ax[0][1].plot(X[:,1])
# ax[0][1].set_title('Phase B')
# ax[1][0].plot(X[:,2])
# ax[1][0].set_title('Phase C')
# ax[1][1].plot(X[:,3])
# ax[1][1].set_title('Z Signal')
# f.suptitle(f'{sample["fault_type"]} fault with {sample["angle"]}°, {sample["distance"]} km '
#            f'and {sample["resistance"].split(".")[0]} Ω.')


# f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
# ax[0][0].plot(S_[:,0])
# ax[0][0].set_title('Component 1')
# ax[0][1].plot(S_[:,1])
# ax[0][1].set_title('Component 2')
# ax[1][0].plot(S_[:,2])
# ax[1][0].set_title('Component 3')
# ax[1][1].plot(S_[:,3])
# ax[1][1].set_title('Component 4')
# f.suptitle(f'ICA components for {sample["fault_type"]} fault with {sample["angle"]}°, {sample["distance"]} km '
#            f'and {sample["resistance"].split(".")[0]} Ω.')
# plt.show()


data = decompress_pickle(INPUT_DATA_PATH + 'cycle_data')
plt.plot(data[10]['v_cycle_1'][320:640])
print(max(data[10]['v_cycle_1'][320:640]))
plt.show()