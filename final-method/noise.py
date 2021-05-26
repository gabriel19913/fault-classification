import scipy.io as sio
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import bz2
import _pickle as cPickle

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
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(power_signal))
    # Noise up the original signal
    noise_signal = original_signal + noise
    return noise_signal

# A B C T >>> falta BCT: [0, 1, 1, 1]
def convert_label(label):
    mlb = MultiLabelBinarizer()
    mlb.fit(['A', 'B', 'C', 'T'])
    return pd.DataFrame(mlb.transform(np.array([label])))

# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'wb') as f: 
        cPickle.dump(data, f, protocol = 5)

def decompress_pickle(file):
    with bz2.BZ2File(file + '.pbz2', 'rb') as f: 
        data = cPickle.load(f)
    return data

if __name__ == '__main__':
    INPUT_DATA_PATH = '../../input-data/'
    DETECTED_DATA_PATH = '../detected_signals/'
    fileNames = np.array((os.listdir(DETECTED_DATA_PATH)))
    i = 0
    data_list = []
    for file in fileNames:
        if (file[-3:] == 'mat') and (file[:3] == 'det'):
            print(f'Working on sample: {i + 1}')
            mat = sio.loadmat(DETECTED_DATA_PATH + file)
            # Gerating zero signal
            v_Z = mat['V_det'][:, 0] + mat['V_det'][:, 1] + mat['V_det'][:, 2]
            i_Z = mat['I_det'][:, 0] + mat['I_det'][:, 1] + mat['I_det'][:, 2]
            # Applying 60dB noise in voltage signal for each phase
            v_noiseA = add_noise(mat['V_det'][:, 0], 60)
            v_noiseB = add_noise(mat['V_det'][:, 1], 60)
            v_noiseC = add_noise(mat['V_det'][:, 2], 60)
            v_noiseZ = add_noise(v_Z, 60)
            v_noise = np.vstack([v_noiseA, v_noiseB, v_noiseC, v_noiseZ])
            # Applying 60dB noise in current signal for each phase
            i_noiseA = add_noise(mat['I_det'][:, 0], 60)
            i_noiseB = add_noise(mat['I_det'][:, 1], 60)
            i_noiseC = add_noise(mat['I_det'][:, 2], 60)
            i_noiseZ = add_noise(i_Z, 60)
            i_noise = np.vstack([i_noiseA, i_noiseB, i_noiseC, i_noiseZ])
            # Getting fault information
            fault_info = file.split('LOC3_')[-1].replace('.mat', '').split('_')
            fault_type = fault_info[0]
            distance = fault_info[1]
            angle = fault_info[2]
            resistance = fault_info[3]
            fault_type_one_hot = convert_label(fault_type)
            
            fault_dict = {'v_noise': v_noise, 'i_noise': i_noise, 'fault_type': fault_type,
                        'fault_type_one_hot': fault_type_one_hot, 'distance': distance,
                        'angle': angle, 'resistance': resistance}
            data_list.append(fault_dict)
            i += 1

    compressed_pickle(INPUT_DATA_PATH + 'noise_data', data_list)
