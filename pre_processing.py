import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import time
from jmespath import search
from tqdm import tqdm, trange
import functools
import operator


def add_noise(original_signal, snr):
    '''
    Equations:
    [1] SNR = Psignal / Pnoise
    [2] SNRdb = Psignal,db - Pnoise,db
    '''
    # Calculate signal power and convert to dB
    power_signal = original_signal ** 2 
    sig_avg_watts = np.mean(power_signal)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - snr
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts),
                             len(power_signal))
    # Noise up the original signal
    noise_signal = original_signal + noise
    return noise_signal


def apply_noise(data):
    final_data = []
    for _, v in enumerate(data):
        fault_dict = {'fault_type': v['fault_type'], 'I_pu': v['I_pu']}
        for snr in range(10, 110, 10):
            noise_a = add_noise(v['I_pu'][:, 0], snr).reshape(-1, 1)
            noise_b = add_noise(v['I_pu'][:, 1], snr).reshape(-1, 1)
            noise_c = add_noise(v['I_pu'][:, 2], snr).reshape(-1, 1)
            noise_signal = np.concatenate((noise_a, noise_b, noise_c), axis=1)
            fault_dict.update({f'I_{str(snr)}db': noise_signal})
        final_data.append(fault_dict)
    return final_data


def norm_data(data):
    lista_I = []
    for item in data:
        lista_I.append(np.max(item['I_nom'][:3000, :], axis=0))
    I_mean = round(np.mean(lista_I), 0)
    I_pu_data = []
    for i, v in enumerate(data):
        I_pu = data[i]['I_det'] / I_mean
        I_pu_data.append({'I_pu': I_pu, 'fault_type': data[i]['faultType']})
    return I_pu_data


def drange(start, stop, step):
    while start < stop:
        yield start
        start *= step


def split_cycles(final_data):
    final_samples = []
    keys = ['I_pu', 'I_10db', 'I_20db', 'I_30db', 'I_40db', 'I_50db', 'I_60db',
            'I_70db', 'I_80db', 'I_90db', 'I_100db']
    for data in final_data:
        append_dict = {}
        for n in drange(1, 33, 2):
            data_dict = {}
            for key in keys:
                size = int((data[key][64:, :].shape[0] - 64) / n + 64)
                signal = data[key][64:, :]
                signal_z = signal.sum(axis=1).reshape(-1, 1)
                final = np.concatenate((signal, signal_z), axis=1)
                data_dict[key] = final[:size, :]
            append_dict.update({'fault_type': data['fault_type'][0],
                                'cycle_' + str(n): data_dict})
        final_samples.append(append_dict)
    return final_samples


def data_without_pre(final_data):
    final_samples = []
    keys = ['I_pu', 'I_10db', 'I_20db', 'I_30db', 'I_40db', 'I_50db', 'I_60db',
            'I_70db', 'I_80db', 'I_90db', 'I_100db']
    for data in final_data:
        append_dict = {}
        for n in drange(1, 33, 2):
            data_dict = {}
            for key in keys: 
                size = int(data[key][128:, :].shape[0] / n)
                signal = data[key][128:, :]
                signal_z = signal.sum(axis=1).reshape(-1, 1)
                final = np.concatenate((signal, signal_z), axis=1)
                data_dict[key] = final[:size, :]
            append_dict.update({'fault_type': data['fault_type'][0],
                                'cycle_' + str(n): data_dict})
        final_samples.append(append_dict)
    return final_samples


def convert_label(final_samples):
    # ! A B C T >>> falta BCT: [0, 1, 1, 1]
    mlb = MultiLabelBinarizer()
    mlb.fit(['A', 'B', 'C', 'T'])
    for data in final_samples:
        label = data['fault_type']
        out_vector = pd.DataFrame(mlb.transform(np.array([label])),
                                  columns=['A', 'B', 'C', 'T'])
        data['fault_type_bin'] = out_vector


def plot_data(sample, data, norm, noisy, save):
    style.use('seaborn-talk')
    style.use('ggplot')
    fig, a = plt.subplots(2, 2, sharex=True, figsize=(15, 10))
    fault = data[sample]['faultType']
    fig.suptitle(f'Falta {fault[0]} - 1/2 pré-falta e 1/2 ciclo pós falta',
                 fontsize=18, fontweight='bold')
    a[0][0].plot(data[sample]['I_det'])
    a[0][0].set_title('Sinal de corrente')
    a[0][0].set_ylabel('I (A)')
    a[0][1].plot(norm[sample]['I_pu'])
    a[0][1].set_title('Sinal de corrente normalizado')
    a[0][1].set_ylabel('I (pu)')
    a[1][0].plot(noisy[sample]['I_10db'])
    a[1][0].set_title('Sinal de corrente com ruído SNR de 10dB')
    a[1][0].set_ylabel('I (pu)')
    a[1][0].set_xlabel('Amostras')
    a[1][1].plot(noisy[sample]['I_30db'])
    a[1][1].set_title('Sinal de corrente com ruído SNR de 30dB')
    a[1][1].set_ylabel('I (pu)')
    a[1][1].set_xlabel('Amostras')
    plt.tight_layout()
    if save:
        fig_path = 'fig/'
        plt.savefig(fig_path + 'current_signal.pdf', dpi=100)
    plt.show()


def save_to_file(path, file_name, data):
    with open(path + file_name + '.pkl', 'wb') as f:
        data = pickle.dump(data, f)


# OBS: Somente as três últimas funções são utilizadas, as primeiras são apenas
# funções auxiliares.

def cum2Calc(vetMediaZero, nPoints, ii):
    return np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii))/nPoints


def cum3Calc(vetMediaZero, nPoints, ii):
    return np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii)**2)/nPoints


def cum4Calc(vetMediaZero, nPoints, ii):
    sumOfSquares = np.dot(vetMediaZero.T, vetMediaZero)
    part1 = np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii)**3)/nPoints
    part2 = 3*np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii))*sumOfSquares/(nPoints**2)
    return part1 - part2


def cumCalc(vetEntrada, nEvents, order):
    if (order == 2):
        functionCalled = cum2Calc
    elif (order == 3):
        functionCalled = cum3Calc
    elif (order == 4):
        functionCalled = cum4Calc
    else:
        return None

    # Transformando vetEntrada em um vetor coluna:
    dimVet = vetEntrada.shape
    if(dimVet[0] == nEvents):
        vetEntrada = vetEntrada.T

    nPoints = vetEntrada[:, 0].size  # number of points per column

    # Pre - allocating space
    cum = np.zeros([nEvents, nPoints])

    for i in range(nEvents):
        # Transformando vetEntrada em um vetor de média nula:
        media = np.mean(vetEntrada[:,i])
        vetMediaZero = vetEntrada[:,i] - media

        for ii in range(nPoints):
            cum[i, ii] = functionCalled(vetMediaZero, nPoints, ii)

    return cum


def cum2(vetEntrada, nEvents):
    return cumCalc(vetEntrada, nEvents, 2)


def cum3(vetEntrada, nEvents):
    return cumCalc(vetEntrada, nEvents, 3)


def cum4(vetEntrada, nEvents):
    return cumCalc(vetEntrada, nEvents, 4)


def apply_hos(data):
    key_list = []
    for key in data[0].keys():
        if 'cycle' in key:
            key_list.append([key + '.' + k for k in data[0]['cycle_1'].keys()])
    key_list = functools.reduce(operator.iconcat, key_list, [])
    timeStart = time.perf_counter()
    types = search("[*].fault_type", data)
    cum_data_dict = {}
    for key in key_list:
        print(f'Starting signal {key}.')
        src = f"[*].{key}"
        signal = search(src, data)

        signal_a_list = []
        signal_b_list = []
        signal_c_list = []
        signal_z_list = []
        for item in signal:
            signal_a_list.append(item[:, 0].reshape(1, -1))
            signal_b_list.append(item[:, 1].reshape(1, -1))
            signal_c_list.append(item[:, 2].reshape(1, -1))
            signal_z_list.append(item[:, 3].reshape(1, -1))
        signal_a = np.row_stack(signal_a_list)
        signal_b = np.row_stack(signal_b_list)
        signal_c = np.row_stack(signal_c_list)
        signal_z = np.row_stack(signal_z_list)

        fundCum2a = cum2(signal_a, 940)
        fundCum3a = cum3(signal_a, 940)
        fundCum4a = cum4(signal_a, 940)
        fundCum2b = cum2(signal_b, 940)
        fundCum3b = cum3(signal_b, 940)
        fundCum4b = cum4(signal_b, 940)
        fundCum2c = cum2(signal_c, 940)
        fundCum3c = cum3(signal_c, 940)
        fundCum4c = cum4(signal_c, 940)
        fundCum2z = cum2(signal_z, 940)
        fundCum3z = cum3(signal_z, 940)
        fundCum4z = cum4(signal_z, 940)

        new_key = '_'.join(src.split('.')[1:])
        cum_dict = {new_key: {
         'cum2': {
             'A':
                 pd.concat([pd.DataFrame(fundCum2a),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             'B':
                 pd.concat([pd.DataFrame(fundCum2b),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             'C':
                 pd.concat([pd.DataFrame(fundCum2c),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             'Z':
                 pd.concat([pd.DataFrame(fundCum2z),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             },
         'cum3': {
             'A':
                 pd.concat([pd.DataFrame(fundCum3a),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             'B':
                 pd.concat([pd.DataFrame(fundCum3b),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             'C':
                 pd.concat([pd.DataFrame(fundCum3c),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             'Z':
                 pd.concat([pd.DataFrame(fundCum3z),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             },
         'cum4': {
             'A':
                 pd.concat([pd.DataFrame(fundCum4a),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             'B':
                 pd.concat([pd.DataFrame(fundCum4b),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             'C':
                 pd.concat([pd.DataFrame(fundCum4c),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             'Z':
                 pd.concat([pd.DataFrame(fundCum4z),
                            pd.Series(types, name='fault_type'),
                            pd.Series(types, name='fault_type_bin')], axis=1),
             },
         }
        }
        cum_data_dict.update(cum_dict)
        print(f'Ending signal {key}.')
    timeElapsed = time.perf_counter() - timeStart
    return cum_data_dict, timeElapsed


def preprocess(file_name, save_flag=False, plot_flag=False):
    data_path = 'data/'
    with open(data_path + 'data_with_current.pkl', 'rb') as f:
        # ! Observação cada sinal tem 384 pontos amostrais
        # ! Meio ciclo pré-falta (128 pontos)
        # ! Um ciclo pós-falta (256 pontos)
        data = pickle.load(f)
    norm = norm_data(data)
    noisy = apply_noise(norm)
    cycle = split_cycles(noisy)
    convert_label(cycle)
    if save_flag:
        save_to_file('data/', file_name + '.pkl', cycle)
    if plot_flag:
        plot_data(50, data, norm, noisy, False)


if __name__ == "__main__":
    data_path = 'data/'
    with open(data_path + 'noisy_data_without_pre.pkl', 'rb') as f:
        data = pickle.load(f)
    cum_data, time_elapsed = apply_hos(data)
    save_to_file(data_path, 'noisy_data_without_pre_cum', cum_data)
    print(f'Time to apply higher order statistics: {time_elapsed}')
