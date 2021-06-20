from noise import decompress_pickle, compressed_pickle
import numpy as np

INPUT_DATA_PATH = '../input-data/'

def drange(start, stop, step):
    while start < stop:
        yield start
        start *= step

def find_max(X):
    max = np.max(X)
    min = np.abs(np.min(X))
    if max > min:
        return max
    else:
        return min

def normalize(X):
    return X / find_max(X)

if __name__ == "__main__":
    all_data = decompress_pickle(INPUT_DATA_PATH + 'noise_data')
    data_list = []
    for i, data in enumerate(all_data):
        print(f'Working on sample: {i + 1}')
        fault_dict = {'fault_type': data['fault_type'],
                      'fault_type_one_hot': data['fault_type_one_hot'],
                      'distance': data['distance'], 'angle': data['angle'],
                      'resistance': data['resistance'], 'v_noise': data['v_noise'],
                      'i_noise': data['i_noise']}
        for n in drange(1, 33, 2):
            size = int((data['i_noise'][0,64:].shape[0] - 64) / n + 64)
            i_signalA = data['i_noise'][0,64:][:size]
            i_signalB = data['i_noise'][1,64:][:size]
            i_signalC = data['i_noise'][2,64:][:size]
            i_signalZ = data['i_noise'][3,64:][:size]
            i_signal = np.hstack([i_signalA, i_signalB, i_signalC, i_signalZ])

            v_signalA = data['v_noise'][0,64:][:size]
            v_signalB = data['v_noise'][1,64:][:size]
            v_signalC = data['v_noise'][2,64:][:size]
            v_signalZ = data['v_noise'][3,64:][:size]
            v_signal = np.hstack([v_signalA, v_signalB, v_signalC, v_signalZ])
            fault_dict.update({f'i_cycle_{n}': i_signal, f'v_cycle_{n}': v_signal})
        data_list.append(fault_dict)
    compressed_pickle(INPUT_DATA_PATH + 'cycle_data', data_list)
