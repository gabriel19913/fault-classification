import re
import numpy as np
import time
from jmespath import search
from tqdm import tqdm
from funcs import gen_key_list
from joblib import Parallel, delayed, parallel_backend

# OBS: Somente as três últimas funções são utilizadas, as primeiras são apenas
# funções auxiliares.


def cum2Calc(vetMediaZero, nPoints, ii):
    return np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii))/nPoints


def cum3Calc(vetMediaZero, nPoints, ii):
    return np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii)**2)/nPoints


def cum4Calc(vetMediaZero, nPoints, ii):
    sumOfSquares = np.dot(vetMediaZero.T, vetMediaZero)
    part1 = np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii)**3)/nPoints
    part2 = 3*np.matmul(vetMediaZero.T, np.roll(vetMediaZero,
                                                ii))*sumOfSquares/(nPoints**2)
    return part1 - part2


def cumCalc(vetEntrada, nEvents, order):
    if (order == 2):
        functionCalled = cum2Calc
    elif (order == 3):
        functionCalled = cum3Calc
    elif (order == 4):
        functionCalled = cum4Calc
    else:
        return

    # Transformando vetEntrada em um vetor coluna:
    dimVet = vetEntrada.shape
    if(dimVet[0] == nEvents):
        vetEntrada = vetEntrada.T

    nPoints = vetEntrada[:, 0].size  # number of points per column

    # Pre - allocating space
    cum = np.zeros([nEvents, nPoints])
    for i in range(nEvents):
        # Transformando vetEntrada em um vetor de média nula:
        media = np.mean(vetEntrada[:, i])
        vetMediaZero = vetEntrada[:, i] - media

        for ii in range(nPoints):
            cum[i, ii] = functionCalled(vetMediaZero, nPoints, ii)

    return cum


def cum2(vetEntrada, nEvents):
    return cumCalc(vetEntrada, nEvents, 2)


def cum3(vetEntrada, nEvents):
    return cumCalc(vetEntrada, nEvents, 3)


def cum4(vetEntrada, nEvents):
    return cumCalc(vetEntrada, nEvents, 4)


def apply_cum(data):
    key_list = gen_key_list(data)
    types = search("[*].fault_type", data)
    types_bin = search("[*].fault_type_bin", data)
    timeStart = time.perf_counter()
    types = search("[*].fault_type", data)
    cum_data_dict = {}
    for key in tqdm(key_list, desc='key_list'):
        src = f"[*].{key}"
        signal = search(src, data)

        signal_a_list = []
        signal_b_list = []
        signal_c_list = []
        signal_z_list = []
        for s in signal:
            signal_a_list.append(s[:, 0].reshape(1, -1))
            signal_b_list.append(s[:, 1].reshape(1, -1))
            signal_c_list.append(s[:, 2].reshape(1, -1))
            signal_z_list.append(s[:, 3].reshape(1, -1))
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
        cum_dict = {new_key:
                    {'cum2': {'A': fundCum2a, 'B': fundCum2b, 'C': fundCum2c,
                              'Z': fundCum2z, 'fault_type': types,
                              'fault_type_bin': types_bin},
                     'cum3': {'A': fundCum3a, 'B': fundCum3b, 'C': fundCum3c,
                              'Z': fundCum3z, 'fault_type': types,
                              'fault_type_bin': types_bin},
                     'cum4': {'A': fundCum4a, 'B': fundCum4b, 'C': fundCum4c,
                              'Z': fundCum4z, 'fault_type': types,
                              'fault_type_bin': types_bin}
                     }
                    }
        cum_data_dict.update(cum_dict)
    timeElapsed = time.perf_counter() - timeStart
    print(f'Elapsed time to apply cumulants in the data {timeElapsed}')
    return cum_data_dict
