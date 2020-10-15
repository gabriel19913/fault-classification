# %%
%load_ext autoreload
%autoreload 2
# %%
from hos.bispectrumd import bispectrumd
from hos.bispectrumi import bispectrumi
from funcs import open_data
import matplotlib.pyplot as plt
from hos.polycoherence import _plot_signal, polycoherence, plot_polycoherence

# %%
model_path = 'models/'
fig_path = 'fig/'

# %%
def plot_fault(signal):
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8))
    plt.plot(signal['cycle_1']['I_pu'])
    plt.legend(['Fase A', 'Fase B', 'Fase C', 'A + B + C'])
    plt.ylabel('I (pu)')
    plt.xlabel('Amostas')
    plt.title(f"Sinal de corrente para uma falta {i_at['fault_type']}")
    plt.tight_layout()
    plt.savefig(fig_path + f"fault_{i_at['fault_type']}" + '.png')
    plt.show()
# %%
v_data = open_data('v_noise_data.pkl')
i_data = open_data('i_noise_data.pkl')
# %%
i_at = i_data[7]
i_bt = i_data[5]
i_ct = i_data[4]
i_ab = i_data[6]
i_ca = i_data[2]
i_bc = i_data[19]
i_abt = i_data[15]
i_bct = i_data[0]
i_cat = i_data[1]
i_abc = i_data[3]
# %%
# plot_fault(i_at)
# # %%
# freq1, fre2, bispec = polycoherence(i_at['cycle_1']['I_pu'][:,3], 15360, norm=None)
# plot_polycoherence(freq1, fre2, bispec)
# # %%
# freq1, fre2, bispec = polycoherence(i_bt['cycle_1']['I_pu'][:,0], 15360, norm=None)
# plot_polycoherence(freq1, fre2, bispec)
# # %%
# freq1, fre2, bispec = polycoherence(i_ct['cycle_1']['I_pu'][:,0], 15360, norm=None)
# plot_polycoherence(freq1, fre2, bispec)
# # %%
# freq1, fre2, bispec = polycoherence(i_ab['cycle_1']['I_pu'][:,0], 15360, norm=None)
# plot_polycoherence(freq1, fre2, bispec)
# %%
signals = [i_at, i_bt, i_ct, i_ab, i_ca, i_bc, i_abt, i_bct, i_cat, i_abc]
for s in signals:
    result = bispectrumi(s['cycle_32']['I_pu'], s['fault_type'], nlag = 5)
# %%
result
# %%

# %%
result = bispectrumi(i_at['cycle_32']['I_pu'], i_at['fault_type'], nlag = 5)
# %%
result = bispectrumi(i_at['cycle_16']['I_100db'], i_at['fault_type'], nlag = 5)
# %%
result = bispectrumi(i_bt['cycle_32']['I_pu'], i_bt['fault_type'], nlag = 5)
# %%
result = bispectrumi(i_ct['cycle_1']['I_pu'], nlag = 5)
#%%
result = bispectrumi(i_ab['cycle_1']['I_pu'], nlag = 5)
#%%
result = bispectrumi(i_ca['cycle_1']['I_pu'], nlag = 5)
# %%
result = bispectrumi(i_bc['cycle_1']['I_pu'], nlag = 5)
# %%
result = bispectrumi(i_abt['cycle_1']['I_pu'], nlag = 5)
# %%
result = bispectrumi(i_bct['cycle_1']['I_pu'], nlag = 5)
# %%
result = bispectrumi(i_cat['cycle_1']['I_pu'], nlag = 5)
# %%
result = bispectrumi(i_abc['cycle_1']['I_pu'], nlag = 5)
result[0].shape
# %%
result[1].shape
# %%

# %%