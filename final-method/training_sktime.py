import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from sktime.transformations.panel.rocket import Rocket, MiniRocketMultivariate
from sklearn.model_selection import StratifiedKFold
import pickle
from glob import glob
import time
from noise import decompress_pickle

INPUT_DATA_PATH = '../input-data/'
MODEL_PATH = './models/'

def open_folds(cycle, train_test, X_y, v_i):
    """
    Parameters:
        cycle      : which cycle, ex.: 'cycle_1' (1, 2, 4, 8, 16, 32)
        train_test : if it is the train ot test set, ex: 'train' (train, test)
        X_y        : if it is the X or y set, ex.: 'X' (X, y)
        v_i        : if it is a voltage or current signal, ex.: 'i' (v, i)
    Return:
        list : each fold is in a position.
    """
    paths = list(map(lambda x: x.split('.pbz2')[0], glob(INPUT_DATA_PATH +
                                                         f'folds/{v_i}/{cycle}/{X_y}_{train_test}_fold_[0-9]*.pbz2')))
    paths.sort(key = lambda x: int(x.split('_')[-1]))
    data_list = []
    for path in paths:
        folder_pos = int(path.split('/')[-1].split('_')[-1]) - 1
        fold = decompress_pickle(path)
        data_list.insert(folder_pos, fold)
    return data_list

def find_max(X):
    max = np.max(X)
    min = np.abs(np.min(X))
    if max > min:
        return max
    else:
        return min
    
def format_dataframe(data):
    cols = int(data.shape[0] / 4)
    shaped_data = data.reshape((4, cols)).T
    s1 = pd.Series(shaped_data[:, 0])
    s2 = pd.Series(shaped_data[:, 1])
    s3 = pd.Series(shaped_data[:, 2])
    s4 = pd.Series(shaped_data[:, 3])
    phases_dict = {'A': [], 'B': [], 'C': [], 'Z': []}
    phases_dict['A'].append(s1)
    phases_dict['B'].append(s2)
    phases_dict['C'].append(s3)
    phases_dict['Z'].append(s4)
    return pd.DataFrame(phases_dict)

def find_max_value(X):
    # Finding max value in by phase in training set to normalize signals
    max_list = [0, 0, 0, 0]
    for k, v in {'A': 0, 'B': 1, 'C': 2, 'Z': 3}.items():
        for row in X[k]:
            max_value = find_max(row)
            if max_value > max_list[v]:
                max_list[v] = max_value
    return max_list

def normalizing(X, max_list):
    X['A'] = X['A'] / max_list[0]
    X['B'] = X['B'] / max_list[1]
    X['C'] = X['C'] / max_list[2]
    X['Z'] = X['Z'] / max_list[3]
    return X

def evaluating_model(model, X_test, y_test, cycle, scores, count, model_name='model'):
    # Evaluating model
    score = model.score(X_test, y_test)
    scores.append(score)
    if len(scores) != 1 and score > scores[count] or len(scores) == 1:
        pickle.dump(model, open(MODEL_PATH + f'{model_name}_{cycle}.pkl', 'wb'))
    return scores

def print_results(scores, end_time, start_time):
    folds_labels = [f'Fold {i}' for i in range(1, 11)]
    print(f'Acurácia em cada fold:')
    for k, v in dict(zip(folds_labels, np.round(scores * 100, decimals=2))).items():
        print(f'{k:<7}: {v:^7}%')
    print(f'\nMédia da acurácia: {np.mean(scores) * 100:.2f}%')
    print(f'Desvio padrão da acurácia: {np.std(scores) * 100:.2f}%')
    print(f'Tempo necessário para treinamento {int(np.round(end_time - start_time, 0))} segundos')

def kfold(train_X, train_y, test_X, test_y, max_list, model, cycle, model_name='',
          transformation=None):
    scores = []
    s = time.time()
    for count, (X_tr, y_tr, X_te, y_te) in enumerate(zip(train_X, train_y, test_X, test_y),
                                                     start=-1):
        X_tr_norm = normalizing(X_tr, max_list)
        X_te_norm = normalizing(X_te, max_list)

        # Transforming data
        if transformation:
            X_tr_transform = transformation.transform(X_tr_norm)
            X_te_transform = transformation.transform(X_te_norm)
        else:
            X_tr_transform = X_tr_norm.copy()
            X_te_transform = X_te_norm.copy()

        model.fit(X_tr_transform, y_tr)
        scores = evaluating_model(model, X_te_transform, y_te, cycle, scores, count, model_name)

    e = time.time()
    final_scores = np.array(scores)
    print_results(final_scores, e, s)

def validating(X_val, y_val, model_name, cycle):
    with open(MODEL_PATH + f'{model_name}_{cycle}.pkl', 'rb') as f:
        best_model = pickle.load(f)
    val_score = best_model.score(X_val, y_val)
    print('*' * 50)
    print(f'Acurácia no conjunto de validação: {val_score * 100:.2f}%')
    print('*' * 50)

def training(signal, cycle, model, model_name='', transformation=None):
    X_train = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal}/{cycle}/X_train')
    X_val = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal}/{cycle}/X_val')
    y_val = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal}/{cycle}/y_val')

    max_list = find_max_value(X_train)
    X_train_norm = normalizing(X_train, max_list)
    X_val_norm = normalizing(X_val, max_list)
    
    if transformation:
        transformation.fit(X_train_norm)
        X_val_transform = transformation.transform(X_val_norm)
    else:
        X_val_transform = X_val_norm.copy()

    # Opening the folds and saving as lists for training and testing 
    train_X = open_folds(cycle, 'train', 'X', signal)
    train_y = open_folds(cycle, 'train', 'y', signal)
    test_X = open_folds(cycle, 'test', 'X', signal)
    test_y = open_folds(cycle, 'test', 'y', signal)
    kfold(train_X, train_y, test_X, test_y, max_list, model, cycle, model_name, transformation)
    validating(X_val_transform, y_val, model_name, cycle)


if __name__ == '__main__':
    num_features = 100
    transformation = MiniRocketMultivariate(num_features=num_features)
    model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    signal, cycle, model_name = 'i', 'cycle_1', 'minirocket'
    training(signal, cycle, model, model_name, transformation)