from noise import decompress_pickle
import pandas as pd
import numpy as np
import pickle
from training_rf_new_dataset import find_max_value, normalizing, transform_tabular_data, apply_notch, notch_filter
import time
INPUT_DATA_PATH = '../input-data/'
MODEL_RF_PATH_NEW = './models_rf_new/'
MODEL_RF_PATH = './models_rf/'
MODEL_ROCKET_PATH_NEW = './models/new_dataset/'
MODEL_ROCKET_PATH = './models/'

def rf_dataset1_time(signal_type, cycle):
    print(f"Tempo operacional tipo do sinal {signal_type}, {cycle} - Random Forest")
    with open(MODEL_RF_PATH + f'{signal_type}_random_forest_{cycle}_max_values.pkl', 'rb') as f:
        max_list = pickle.load(f)

    X_val = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/X_val')
    X_val = transform_tabular_data(X_val)
    y_val = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/y_val')
    X_val['target'] = y_val
    sample = X_val.groupby('target').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
    
    letter_dict = {'A': 0, 'B': 0, 'C': 0, 'Z': 0}
    for i in sample.drop(columns='target').columns:
        letter = i.split('_')[0]
        letter_dict[letter] += 1
    time_dict = {}
    for i in range(10):
        s = pd.DataFrame(sample.iloc[i,:-1]).T
        target = sample.loc[i, 'target']
        v = letter_dict['A']
        v2 = 2*v
        v3 = 3*v
        index = int(s.shape[1] / 4)
        # time
        start = time.time()
        a = s.iloc[:,:v] / max_list[0]
        b = s.iloc[:,v:v2] / max_list[1]
        c = s.iloc[:,v2:v3] / max_list[2]
        z = s.iloc[:,v3:] / max_list[3]
        one_fault_sample = pd.concat([a,b,c,z], axis=1)
        one_fault_sample = apply_notch(notch_filter, one_fault_sample, index)
        with open(MODEL_RF_PATH + f'{signal_type}_random_forest_classifier_{cycle}.pkl', 'rb') as f:
            best_model = pickle.load(f)
        y_pred = best_model.predict(one_fault_sample)
        end = time.time()
        time_dict[target] = end - start
        # print(f"Print o tempo operacional para uma amostra da falta {target} foi {end-start}s.")
    max_time = max(time_dict, key=time_dict.get)
    min_time = min(time_dict, key=time_dict.get)
    print(f"O tipo de falta com maior tempo operacional é a falta {max_time}, com {time_dict[max_time]:.3f}s.")
    print(f"O tipo de falta com menor tempo operacional é a falta {min_time}, com {time_dict[min_time]:.3f}s.")

def rf_dataset2_time(signal_type, cycle):
    print(f"Tempo operacional tipo do sinal {signal_type}, {cycle} - Random Forest")
    with open(MODEL_RF_PATH_NEW + f'{signal_type}_random_forest_{cycle}_max_values.pkl', 'rb') as f:
        max_list = pickle.load(f)
    X_val_flavio = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/X_val')
    y_val_flavio = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/y_val')
    X_val_robson = decompress_pickle(INPUT_DATA_PATH + f'folds-robson/{signal_type}/{cycle}/X_val')
    y_val_robson = decompress_pickle(INPUT_DATA_PATH + f'folds-robson/{signal_type}/{cycle}/y_val')
    X_val = pd.concat([X_val_flavio, X_val_robson]).reset_index(drop=True)
    X_val = transform_tabular_data(X_val)
    y_val = np.concatenate([y_val_flavio, y_val_robson])

    X_val['target'] = y_val
    sample = X_val.groupby('target').apply(lambda x: x.sample(n=1)).reset_index(drop=True)

    letter_dict = {'A': 0, 'B': 0, 'C': 0, 'Z': 0}
    for i in sample.drop(columns='target').columns:
        letter = i.split('_')[0]
        letter_dict[letter] += 1
        
    time_dict = {}
    for i in range(10):
        s = pd.DataFrame(sample.iloc[i,:-1]).T
        target = sample.loc[i, 'target']
        v = letter_dict['A']
        v2 = 2*v
        v3 = 3*v
        index = int(s.shape[1] / 4)
        # time
        start = time.time()
        a = s.iloc[:,:v] / max_list[0]
        b = s.iloc[:,v:v2] / max_list[1]
        c = s.iloc[:,v2:v3] / max_list[2]
        z = s.iloc[:,v3:] / max_list[3]
        one_fault_sample = pd.concat([a,b,c,z], axis=1)
        one_fault_sample = apply_notch(notch_filter, one_fault_sample, index)
        with open(MODEL_RF_PATH_NEW + f'{signal_type}_random_forest_classifier_{cycle}.pkl', 'rb') as f:
            best_model = pickle.load(f)
        y_pred = best_model.predict(one_fault_sample)
        end = time.time()
        time_dict[target] = end - start
        # print(f"Print o tempo operacional para uma amostra da falta {target} foi {end-start}s.")
    max_time = max(time_dict, key=time_dict.get)
    min_time = min(time_dict, key=time_dict.get)
    print(f"O tipo de falta com maior tempo operacional é a falta {max_time}, com {time_dict[max_time]:.3f}s.")
    print(f"O tipo de falta com menor tempo operacional é a falta {min_time}, com {time_dict[min_time]:.3f}s.")

def rocket_dataset1_time(transformer_name, features, cycle, signal_type):
    print(f"Tempo operacional para {transformer_name} com {features} e {cycle}")
    with open(MODEL_ROCKET_PATH + f'{transformer_name}_{cycle}_max_values_{features}.pkl', 'rb') as f:
        max_list = pickle.load(f)
    with open(MODEL_ROCKET_PATH + f'{transformer_name}_classifier_{cycle}_{features}.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open(MODEL_ROCKET_PATH + f'{transformer_name}_{cycle}_{features}.pkl', 'rb') as f:
        transformer = pickle.load(f)

    X_val = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/X_val')
    X_val = transform_tabular_data(X_val)
    y_val = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/y_val')
    X_val['target'] = y_val
    sample = X_val.groupby('target').apply(lambda x: x.sample(n=1)).reset_index(drop=True)

    letter_dict = {'A': 0, 'B': 0, 'C': 0, 'Z': 0}
    for i in sample.drop(columns='target').columns:
        letter = i.split('_')[0]
        letter_dict[letter] += 1

    time_dict = {}
    for i in range(10):
        s = pd.DataFrame(sample.iloc[i,:-1]).T
        target = sample.loc[i, 'target']
        v = letter_dict['A']
        v2 = 2*v
        v3 = 3*v
        index = int(s.shape[1] / 4)
        # time
        start = time.time()
        a = (s.iloc[:,:v] / max_list[0]).astype(float)
        b = (s.iloc[:,v:v2] / max_list[1]).astype(float)
        c = (s.iloc[:,v2:v3] / max_list[2]).astype(float)
        z = (s.iloc[:,v3:] / max_list[3]).astype(float)

        a = pd.Series(a.values.flatten())
        b = pd.Series(b.values.flatten())
        c = pd.Series(c.values.flatten())
        z = pd.Series(z.values.flatten())

        dicio = {'A': [], 'B': [], 'C': [], 'Z': []}
        dicio['A'].append(a)
        dicio['B'].append(b)
        dicio['C'].append(c)
        dicio['Z'].append(z)
        new_sample = pd.DataFrame(dicio)
        new_sample = transformer.transform(new_sample)
        y_pred = classifier.predict(new_sample)
        end = time.time()
        time_dict[target] = end - start
    max_time = max(time_dict, key=time_dict.get)
    min_time = min(time_dict, key=time_dict.get)
    print(f"O tipo de falta com maior tempo operacional é a falta {max_time}, com {time_dict[max_time]:.3f}s.")
    print(f"O tipo de falta com menor tempo operacional é a falta {min_time}, com {time_dict[min_time]:.3f}s.")

def rocket_dataset2_time(transformer_name, features, cycle, signal_type):
    print(f"Tempo operacional para {transformer_name} com {features} e {cycle}")
    with open(MODEL_ROCKET_PATH + f'{transformer_name}_{cycle}_max_values_{features}.pkl', 'rb') as f:
        max_list = pickle.load(f)
    with open(MODEL_ROCKET_PATH + f'{transformer_name}_classifier_{cycle}_{features}.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open(MODEL_ROCKET_PATH + f'{transformer_name}_{cycle}_{features}.pkl', 'rb') as f:
        transformer = pickle.load(f)

    X_val_flavio = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/X_val')
    y_val_flavio = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/y_val')
    X_val_robson = decompress_pickle(INPUT_DATA_PATH + f'folds-robson/{signal_type}/{cycle}/X_val')
    y_val_robson = decompress_pickle(INPUT_DATA_PATH + f'folds-robson/{signal_type}/{cycle}/y_val')
    X_val = pd.concat([X_val_flavio, X_val_robson]).reset_index(drop=True)
    X_val = transform_tabular_data(X_val)
    y_val = np.concatenate([y_val_flavio, y_val_robson])

    X_val['target'] = y_val
    sample = X_val.groupby('target').apply(lambda x: x.sample(n=1)).reset_index(drop=True)

    letter_dict = {'A': 0, 'B': 0, 'C': 0, 'Z': 0}
    for i in sample.drop(columns='target').columns:
        letter = i.split('_')[0]
        letter_dict[letter] += 1

    time_dict = {}
    for i in range(10):
        s = pd.DataFrame(sample.iloc[i,:-1]).T
        target = sample.loc[i, 'target']
        v = letter_dict['A']
        v2 = 2*v
        v3 = 3*v
        index = int(s.shape[1] / 4)
        # time
        start = time.time()
        a = (s.iloc[:,:v] / max_list[0]).astype(float)
        b = (s.iloc[:,v:v2] / max_list[1]).astype(float)
        c = (s.iloc[:,v2:v3] / max_list[2]).astype(float)
        z = (s.iloc[:,v3:] / max_list[3]).astype(float)

        a = pd.Series(a.values.flatten())
        b = pd.Series(b.values.flatten())
        c = pd.Series(c.values.flatten())
        z = pd.Series(z.values.flatten())

        dicio = {'A': [], 'B': [], 'C': [], 'Z': []}
        dicio['A'].append(a)
        dicio['B'].append(b)
        dicio['C'].append(c)
        dicio['Z'].append(z)
        new_sample = pd.DataFrame(dicio)
        new_sample = transformer.transform(new_sample)
        y_pred = classifier.predict(new_sample)
        end = time.time()
        time_dict[target] = end - start
    max_time = max(time_dict, key=time_dict.get)
    min_time = min(time_dict, key=time_dict.get)
    print(f"O tipo de falta com maior tempo operacional é a falta {max_time}, com {time_dict[max_time]:.3f}s.")
    print(f"O tipo de falta com menor tempo operacional é a falta {min_time}, com {time_dict[min_time]:.3f}s.")

if __name__ == '__main__':
    cycles = ['cycle_1', 'cycle_2', 'cycle_4', 'cycle_8', 'cycle_16', 'cycle_32', 'cycle_64', 'cycle_128']
    # Método B
    for cycle in cycles:
        rf_dataset1_time('v', cycle)
    print('='*20)
    # Método C
    for cycle in cycles:
        rf_dataset2_time('v', cycle)
    print('='*20)
    # Método D
    for cycle in cycles:
        rf_dataset1_time('i', cycle)
    print('='*20)
    # Método E
    for cycle in cycles:
        rf_dataset2_time('i', cycle)
    # Método F
    rocket_dataset1_time('rocket', '300', 'cycle_1', 'i')
    rocket_dataset1_time('rocket', '400', 'cycle_2', 'i')
    rocket_dataset1_time('rocket', '400', 'cycle_4', 'i')
    rocket_dataset1_time('rocket', '500', 'cycle_8', 'i')
    rocket_dataset1_time('rocket', '900', 'cycle_16', 'i')
    rocket_dataset1_time('rocket', '1000', 'cycle_32', 'i')
    rocket_dataset1_time('rocket', '900', 'cycle_64', 'i')
    rocket_dataset1_time('rocket', '600', 'cycle_128', 'i')
    rocket_dataset1_time('minirocket', '100', 'cycle_1', 'i')
    rocket_dataset1_time('minirocket', '300', 'cycle_2', 'i')
    rocket_dataset1_time('minirocket', '400', 'cycle_4', 'i')
    rocket_dataset1_time('minirocket', '400', 'cycle_8', 'i')
    rocket_dataset1_time('minirocket', '300', 'cycle_16', 'i')
    rocket_dataset1_time('minirocket', '500', 'cycle_32', 'i')
    rocket_dataset1_time('minirocket', '300', 'cycle_64', 'i')
    rocket_dataset1_time('minirocket', '800', 'cycle_128', 'i')
    print('='*20)
    # Método G
    rocket_dataset2_time('rocket', '200', 'cycle_1', 'i')
    rocket_dataset2_time('rocket', '400', 'cycle_2', 'i')
    rocket_dataset2_time('rocket', '500', 'cycle_4', 'i')
    rocket_dataset2_time('rocket', '500', 'cycle_8', 'i')
    rocket_dataset2_time('rocket', '800', 'cycle_16', 'i')
    rocket_dataset2_time('rocket', '800', 'cycle_32', 'i')
    rocket_dataset2_time('rocket', '700', 'cycle_64', 'i')
    rocket_dataset2_time('rocket', '1000', 'cycle_128', 'i')
    rocket_dataset2_time('minirocket', '500', 'cycle_1', 'i')
    rocket_dataset2_time('minirocket', '500', 'cycle_2', 'i')
    rocket_dataset2_time('minirocket', '500', 'cycle_4', 'i')
    rocket_dataset2_time('minirocket', '900', 'cycle_8', 'i')
    rocket_dataset2_time('minirocket', '700', 'cycle_16', 'i')
    rocket_dataset2_time('minirocket', '600', 'cycle_32', 'i')
    rocket_dataset2_time('minirocket', '600', 'cycle_64', 'i')
    rocket_dataset2_time('minirocket', '500', 'cycle_128', 'i')
    print('='*20)
