# %%
from hos.bispectrumd import bispectrumd
import scipy.io as sio
import matplotlib.pyplot as plt


qpc = sio.loadmat('hos/qpc.mat')
dbic = bispectrumd(qpc['zmat'], 128,3,64,0)
print(dbic[0].shape)
print(dbic[0])
print('=' * 15)
print(dbic[1].shape)
print(dbic[1])
# %%
