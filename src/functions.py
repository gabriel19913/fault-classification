import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import scipy.io as sio
import os


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    # Function by hitvoice
    # https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
    y_true:    true label of the data, with shape (nsamples,)
    y_pred:    prediction of the data, with shape (nsamples,)
    filename:  filename of figure file to save
    labels:    string array, name the order of class labels in the confusion
               matrix. use `clf.classes_` if using scikit-learn models.
               with shape (nclass,).
    ymap:      dict: any -> string, length == nclass.
                if not None, map the labels & ys to more understandable
                strings.
                Caution: original y_true, y_pred and labels must align.
    figsize:   the size of the figure plotted.
    """
    sns.set_context("notebook")
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Real'
    cm.columns.name = 'Predito'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='Greys')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def join_signal(n, data, key):
    final_data = []
    final_class = []
    # 1 ciclo pós falta
    if n == 1:
        for item in data:
            final_data.append(np.hstack([item[key][:, 0], item[key][:, 1],
                                         item[key][:, 2]]))
            final_class.append(item['faultType'][0])

    # 1/2 ciclo pós falta
    if n == 2:
        for item in data:
            final_data.append(np.hstack([item[key][:192, 0], item[key][:192, 1], 
                                         item[key][:192, 2]]))
            final_class.append(item['faultType'][0])

    # 1/4 ciclo pós falta
    if n == 4:
        for item in data:
            final_data.append(np.hstack([item[key][:128, 0], item[key][:128, 1], 
                                         item[key][:128, 2]]))
            final_class.append(item['faultType'][0])

    # 1/8 ciclo pós falta
    if n == 8:
        for item in data:
            final_data.append(np.hstack([item[key][:96, 0], item[key][:96, 1], 
                                         item[key][:96, 2]]))
            final_class.append(item['faultType'][0])

    # 1/16 ciclo pós falta
    if n == 16:
        for item in data:
            final_data.append(np.hstack([item[key][:80, 0], item[key][:80, 1],
                                         item[key][:80, 2]]))
            final_class.append(item['faultType'][0])

    # 1/32 ciclo pós falta
    if n == 32:
        for item in data:
            final_data.append(np.hstack([item[key][:72, 0], item[key][:72, 1], 
                                         item[key][:72, 2]]))
            final_class.append(item['faultType'][0])

    df = pd.DataFrame(np.row_stack(final_data))
    df = pd.concat([df, pd.Series(final_class)], axis=1)
    return df


def param_optimization(X, y, model, grid, n_iter=50, cv=5):
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=grid,
                                   n_iter=n_iter, cv=cv, verbose=2,
                                   n_jobs=-1)
    rf_random.fit(X, y)
    return rf_random.best_params_


def train_model(X_tr, y_tr, model, model_path, cycles, cv=10):
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    scores = []
    final_scores = []
    past_score = 0

    for train_index, test_index in skf.split(X_tr, y_tr):
        X_train, X_test = X_tr.iloc[train_index], X_tr.iloc[test_index]
        y_train, y_test = y_tr.iloc[train_index], y_tr.iloc[test_index]
        for _ in range(100):
            model.fit(X_train, y_train)
            y_pred_1_ciclo = model.predict(X_test)
            score = accuracy_score(y_pred_1_ciclo, y_test)
            scores.append(score)
            if score > past_score:
                print(f'A acurácia atual é {100 * score:.2f}, a acurácia '
                      f'passada era {100 * past_score:.2f}.')
                file_name = f'model_{cycles}_ciclo.joblib'
                with open(model_path + file_name, 'wb') as f:
                    joblib.dump(model, f, compress=('lz4', 3))
                past_score = score
        final_scores.append(np.mean(scores))
    return final_scores


def gen_report(params, scores, tempo, cycles):
    with open("report.txt", "a") as file:
        if cycles > 1:
            file.write(f"================== RELATÓRIO 1/{cycles} CICLO PÓS "
                       "FALTA ==================\n")
        else:
            file.write("================== RELATÓRIO 1 CICLO PÓS FALTA "
                       "==================\n")
        file.write("Model parameters:\n")
        for k, v in params.items():
            file.write(f'\t{k}: {v}\n')
        file.write("\nAcurácia média em cada um dos folds após repetição de "
                   "100 vezes:\n")
        for counter, value in enumerate(scores):
            file.write(f'\tFold {counter + 1}: {value * 100:.2f}%\n')
        file.write(f'\nMédia da acurácia: {np.mean(scores) * 100:.2f}%')
        file.write(f'\nDesvio padrão da acurácia: {np.std(scores) * 100:.2f}%')
        t = time.gmtime(tempo)
        conv_time = time.strftime("%H:%M:%S", t)
        file.write('\nTempo necessário para treinamento do modelo: '
                   f'{conv_time}\n\n')


def save_confusion_matrix(X, y, model_path, fig_path, cycles):
    model_name = model_path + f'model_{cycles}_ciclo.joblib'
    fig_name = fig_path + f'cm_{cycles}_ciclo.pdf'
    with open(model_name, 'rb') as f:
        model = joblib.load(f)
    y_pred = model.predict(X)
    cm_analysis(y.values, y_pred, fig_name, model.classes_, figsize=(10, 10))


def plot_data(data):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax = plt.axes()
    sns.lineplot(data=data['sinal_notch_final'][:, 0])
    sns.lineplot(data=data['sinal_notch_final'][:, 1])
    sns.lineplot(data=data['sinal_notch_final'][:, 2])
    ax.set_title('Saída do filtro notch com 1/4 de pré-falta e 1 cilo pós '
                 f'falta para uma falta {data["faultType"][0]}',
                 fontsize=14, pad=30)
    ax.set_ylabel(f'Saída Filtro Notch (pu)')
    ax.set_xlabel(f'Amostra')
    plt.legend(loc='lower right', labels=['A', 'B', 'C'])
    plt.show()


def data_list(data_path):
    fileNames = np.array((os.listdir(data_path)))
    return [sio.loadmat(data_path + name) for name in fileNames]


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


def apply_noise(data, min=10, max=100, s='I'):
    final_data = []
    for i, v in enumerate(data):
        fault_dict = {'faultType': v['faultType'], f'{s}_pu': v[f'{s}_pu']}
        for snr in range(min, max + 10, 10):
            n0 = add_noise(v[f'{s}_pu'][:, 0], snr)
            n1 = add_noise(v[f'{s}_pu'][:, 1], snr)
            n2 = add_noise(v[f'{s}_pu'][:, 2], snr)
            noise_signal = np.vstack((n0, n1, n2)).T
            fault_dict.update({f'I_{str(snr)}db': noise_signal})
        final_data.append(fault_dict)
    return final_data


def drange(start, stop, step):
    while start < stop:
        yield start
        start *= step


def convert_label(label):
    # A B C T >>> falta BCT: [0, 1, 1, 1]
    mlb = MultiLabelBinarizer()
    mlb.fit(['A', 'B', 'C', 'T'])
    return mlb.transform(np.array(label))[0]


def load_data(path):
    data = pd.read_hdf(path, 'data')
    fault = pd.read_hdf(path, 'fault')
    return data, fault


def save_data(data, s, source='current'):
    noise_path = '../data/noise_signals/'
    keys = data.keys()
    for key in keys:
        if 'fault' not in key:
            for n in drange(1, 65, 2):
                size = int((data[key][64:, :].shape[0] - 64) / n + 64)
                signal = data[key][64:, :]
                signal_z = signal.sum(axis=1).reshape(-1, 1)
                final = np.concatenate((signal, signal_z), axis=1)
                final_signal = pd.DataFrame(final[:size, :],
                                            columns=['A', 'B', 'C', 'Z'])
                cycle_path = f'cycle_{n}/'
                snr_path = f'{key}/{source}/'
                full_path = noise_path + cycle_path + snr_path
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                label = data['faultType']
                out_vector = pd.Series(convert_label(label),
                                       index=['A', 'B', 'C', 'T'],
                                       name='fault_type')
                final_signal.to_hdf(full_path + f'sample_{s + 1}.hdf5',
                                    key='data', mode='w', complevel=5)
                out_vector.to_hdf(full_path + f'sample_{s + 1}.hdf5',
                                  key='fault', complevel=5)
    print(f'Data saved in disk for sample {s + 1}.')


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = []
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles += getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles
