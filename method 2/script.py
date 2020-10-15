import joblib
import matplotlib.pyplot as plt
from scipy.stats.stats import mode
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
import time


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


def join_signal(n, data):
    final_data = []
    final_class = []
    # 1 ciclo pós falta
    if n == 1:
        for item in data:
            final_data.append(np.hstack([item['sinal_notch_final'][:, 0],
                                         item['sinal_notch_final'][:, 1],
                                         item['sinal_notch_final'][:, 2]]))
            final_class.append(item['faultType'][0])

    # 1/2 ciclo pós falta
    if n == 2:
        for item in data:
            final_data.append(np.hstack([item['sinal_notch_final'][:192, 0],
                                         item['sinal_notch_final'][:192, 1],
                                         item['sinal_notch_final'][:192, 2]]))
            final_class.append(item['faultType'][0])

    # 1/4 ciclo pós falta
    if n == 4:
        for item in data:
            final_data.append(np.hstack([item['sinal_notch_final'][:128, 0],
                                         item['sinal_notch_final'][:128, 1],
                                         item['sinal_notch_final'][:128, 2]]))
            final_class.append(item['faultType'][0])

    # 1/8 ciclo pós falta
    if n == 8:
        for item in data:
            final_data.append(np.hstack([item['sinal_notch_final'][:96, 0],
                                         item['sinal_notch_final'][:96, 1],
                                         item['sinal_notch_final'][:96, 2]]))
            final_class.append(item['faultType'][0])

    # 1/16 ciclo pós falta
    if n == 16:
        for item in data:
            final_data.append(np.hstack([item['sinal_notch_final'][:80, 0],
                                         item['sinal_notch_final'][:80, 1],
                                         item['sinal_notch_final'][:80, 2]]))
            final_class.append(item['faultType'][0])

    # 1/32 ciclo pós falta
    if n == 32:
        for item in data:
            final_data.append(np.hstack([item['sinal_notch_final'][:72, 0],
                                         item['sinal_notch_final'][:72, 1],
                                         item['sinal_notch_final'][:72, 2]]))
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


data_path = '../data/detected_signals/'
model_path = 'models/'
fig_path = 'figs/'
fileNames = np.array((os.listdir(data_path)))
teste_data = sio.loadmat(data_path + fileNames[0])

data_list = [sio.loadmat(data_path + name) for name in fileNames]
data_1_ciclo = join_signal(1, data_list)
data_1_2_ciclo = join_signal(2, data_list)
data_1_4_ciclo = join_signal(4, data_list)
data_1_8_ciclo = join_signal(8, data_list)
data_1_16_ciclo = join_signal(16, data_list)
data_1_32_ciclo = join_signal(32, data_list)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num=5)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Division criterion
criterion = ['entropy', 'gini']
# Create the random grid
random_grid = {'criterion': criterion,
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestClassifier()

#* Random Forest
#! 1 ciclo

X_tr_1_ciclo, X_val_1_ciclo, \
    y_tr_1_ciclo, y_val_1_ciclo = train_test_split(data_1_ciclo.iloc[:, :960],
                                                   data_1_ciclo.iloc[:, -1],
                                                   test_size=0.1,
                                                   random_state=42)

model1_best = param_optimization(X_tr_1_ciclo, y_tr_1_ciclo, rf, random_grid)

# Treinamento
model1 = RandomForestClassifier(**model1_best)
s = time.time()
final_scores = train_model(X_tr_1_ciclo, y_tr_1_ciclo, model1, model_path, 1)
e = time.time()
tempo = e - s
final_scores = np.array(final_scores)
print(f'\nMédia da acurácia: {np.mean(final_scores) * 100:.2f}%')
print(f'Desvio padrão da acurácia: {np.std(final_scores) * 100:.2f}%)')

gen_report(model1_best, final_scores, tempo, cycles=1)
save_confusion_matrix(X_val_1_ciclo, y_val_1_ciclo, model_path, fig_path,
                      cycles=1)


#! 1/2 ciclo
X_tr_1_2_ciclo, X_val_1_2_ciclo, \
    y_tr_1_2_ciclo, y_val_1_2_ciclo = train_test_split(data_1_2_ciclo.iloc[:, :576],
                                                       data_1_2_ciclo.iloc[:, -1],
                                                       test_size=0.1, random_state=42)

model2_best = param_optimization(X_tr_1_2_ciclo, y_tr_1_2_ciclo, rf, random_grid)
# Treinamento
model2 = RandomForestClassifier(**model2_best)
s = time.time()
final_scores = train_model(X_tr_1_2_ciclo, y_tr_1_2_ciclo, model2, model_path, 2)
e = time.time()
tempo = e - s
final_scores = np.array(final_scores)
print(f'\nMédia da acurácia: {np.mean(final_scores) * 100:.2f}%')
print(f'Desvio padrão da acurácia: {np.std(final_scores) * 100:.2f}%)')

gen_report(model2_best, final_scores, tempo, cycles=2)
save_confusion_matrix(X_val_1_2_ciclo, y_val_1_2_ciclo, model_path, fig_path,
                      cycles=2)


#! 1/4 ciclo
X_tr_1_4_ciclo, X_val_1_4_ciclo, \
    y_tr_1_4_ciclo, y_val_1_4_ciclo = train_test_split(data_1_4_ciclo.iloc[:, :384],
                                                       data_1_4_ciclo.iloc[:, -1],
                                                       test_size=0.1, random_state=42)

model3_best = param_optimization(X_tr_1_4_ciclo, y_tr_1_4_ciclo, rf, random_grid)
# Treinamento
model3 = RandomForestClassifier(**model3_best)
s = time.time()
final_scores = train_model(X_tr_1_4_ciclo, y_tr_1_4_ciclo, model3, model_path, 4)
e = time.time()
tempo = e - s
final_scores = np.array(final_scores)
print(f'\nMédia da acurácia: {np.mean(final_scores) * 100:.2f}%')
print(f'Desvio padrão da acurácia: {np.std(final_scores) * 100:.2f}%)')

gen_report(model3_best, final_scores, tempo, cycles=4)
save_confusion_matrix(X_val_1_4_ciclo, y_val_1_4_ciclo, model_path, fig_path,
                      cycles=4)


#! 1/8 ciclo
X_tr_1_8_ciclo, X_val_1_8_ciclo, \
    y_tr_1_8_ciclo, y_val_1_8_ciclo = train_test_split(data_1_8_ciclo.iloc[:, :288],
                                                       data_1_8_ciclo.iloc[:, -1],
                                                       test_size=0.1, random_state=42)

model4_best = param_optimization(X_tr_1_8_ciclo, y_tr_1_8_ciclo, rf, random_grid)
# Treinamento
model4 = RandomForestClassifier(**model4_best)
s = time.time()
final_scores = train_model(X_tr_1_8_ciclo, y_tr_1_8_ciclo, model4, model_path, 8)
e = time.time()
tempo = e - s
final_scores = np.array(final_scores)
print(f'\nMédia da acurácia: {np.mean(final_scores) * 100:.2f}%')
print(f'Desvio padrão da acurácia: {np.std(final_scores) * 100:.2f}%)')

gen_report(model4_best, final_scores, tempo, cycles=8)
save_confusion_matrix(X_val_1_8_ciclo, y_val_1_8_ciclo, model_path, fig_path,
                      cycles=8)

#! 1/16 ciclo
X_tr_1_16_ciclo, X_val_1_16_ciclo, \
    y_tr_1_16_ciclo, y_val_1_16_ciclo = train_test_split(data_1_16_ciclo.iloc[:, :240],
                                                         data_1_16_ciclo.iloc[:, -1],
                                                         test_size=0.1, random_state=42)

model5_best = param_optimization(X_tr_1_16_ciclo, y_tr_1_16_ciclo, rf, random_grid)
# Treinamento
model5 = RandomForestClassifier(**model5_best)
s = time.time()
final_scores = train_model(X_tr_1_16_ciclo, y_tr_1_16_ciclo, model5, model_path, 16)
e = time.time()
tempo = e - s
final_scores = np.array(final_scores)
print(f'\nMédia da acurácia: {np.mean(final_scores) * 100:.2f}%')
print(f'Desvio padrão da acurácia: {np.std(final_scores) * 100:.2f}%)')

gen_report(model5_best, final_scores, tempo, cycles=16)
save_confusion_matrix(X_val_1_16_ciclo, y_val_1_16_ciclo, model_path, fig_path,
                      cycles=16)

#! 1/32 ciclo
X_tr_1_32_ciclo, X_val_1_32_ciclo, \
    y_tr_1_32_ciclo, y_val_1_32_ciclo = train_test_split(data_1_32_ciclo.iloc[:, :216],
                                                         data_1_32_ciclo.iloc[:, -1],
                                                         test_size=0.1, random_state=42)

model6_best = param_optimization(X_tr_1_32_ciclo, y_tr_1_32_ciclo, rf, random_grid)
# Treinamento
model6 = RandomForestClassifier(**model6_best)
s = time.time()
final_scores = train_model(X_tr_1_32_ciclo, y_tr_1_32_ciclo, model6, model_path, 32)
e = time.time()
tempo = e - s
final_scores = np.array(final_scores)
print(f'\nMédia da acurácia: {np.mean(final_scores) * 100:.2f}%')
print(f'Desvio padrão da acurácia: {np.std(final_scores) * 100:.2f}%)')

gen_report(model6_best, final_scores, tempo, cycles=32)
save_confusion_matrix(X_val_1_32_ciclo, y_val_1_32_ciclo, model_path, fig_path,
                      cycles=32)
