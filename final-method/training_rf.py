from noise import decompress_pickle
from sklearn.model_selection import RandomizedSearchCV
import time
import numpy as np
from training_sktime import open_folds, find_max_value, normalizing, print_results
from  sktime.transformations.panel.reduce import Tabularizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
from functools import partial
from scipy import signal as sig
import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff


INPUT_DATA_PATH = '../input-data/'
MODEL_PATH = './models_rf/'

def param_optimization(X, y, model, grid, n_iter=50, cv=5):
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=grid,
                                   n_iter=n_iter, cv=cv, verbose=0,
                                   n_jobs=-1)
    rf_random.fit(X, y)
    return rf_random.best_params_

def transform_tabular_data(data, is_list=False):
    t= Tabularizer()
    if is_list:
        return list(map(t.fit_transform, data))
    else:
        return t.fit_transform(data)

def evaluating_model(model, X_test, y_test, cycle, scores, count, max_list, signal_type='',
                     model_name='model', save=None):
    # Evaluating model
    score = model.score(X_test, y_test)
    scores.append(score)
    if save and (
        len(scores) != 1 and score > scores[count] or len(scores) == 1
    ):
        pickle.dump(model, open(MODEL_PATH + f'{signal_type}_{model_name}_classifier_{cycle}.pkl', 'wb'))
        pickle.dump(max_list, open(MODEL_PATH + f'{signal_type}_{model_name}_{cycle}_max_values.pkl', 'wb'))
    return scores

def kfold(train_X, train_y, test_X, test_y, model, cycle, max_list, signal_type, model_name='', save=None):
    scores = []
    s = time.time()
    for count, (X_tr, y_tr, X_te, y_te) in enumerate(zip(train_X, train_y, test_X, test_y),
                                                     start=-1):
        model.fit(X_tr, y_tr)
        scores = evaluating_model(model, X_te, y_te, cycle, scores,
                                  count, max_list, signal_type, model_name, save)
    e = time.time()
    final_scores = np.array(scores)
    print_results(cycle, model_name, final_scores, e, s, save)
    return np.mean(scores) * 100, np.round(e - s, 3)

def ref_grid():
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
    return rf, random_grid

def notch_filter(entrada, fs, f0, fatorNotch):
  w0 = 2 * np.pi * f0 / fs
  a0 = -2 * np.cos(w0)
  B = np.array([1, a0, 1])
  A = np.array([1, a0*fatorNotch, fatorNotch**2])
  entrada1 = np.hstack((entrada, entrada, entrada, entrada, entrada))
  saida1 = sig.lfilter(B, A, entrada1)
  lenSaida1 = saida1.size
  lenEntrada = entrada.size
  sinalTrans = saida1[lenSaida1-lenEntrada:lenSaida1]
  sinalFund  = entrada - sinalTrans
  return sinalFund, sinalTrans

def apply_notch(notch_filter, data, index):
    samples = []
    for i in range(len(data)):
        phase_A = data.iloc[i, 0:index].values
        phase_B = data.iloc[i, index:2*index]
        phase_C = data.iloc[i, 2*index:3*index]
        phase_Z = data.iloc[i, 3*index:4*index]
        _, sinalA = notch_filter(phase_A, 15360, 60, 0.97)
        _, sinalB = notch_filter(phase_B, 15360, 60, 0.97)
        _, sinalC = notch_filter(phase_C, 15360, 60, 0.97)
        _, sinalZ = notch_filter(phase_Z, 15360, 60, 0.97)
        sample = np.concatenate((sinalA, sinalB, sinalC, sinalZ))
        samples.append(sample)
    return pd.DataFrame(samples)

def validating(X_val, y_val, model, model_name, signal_type='', save=None):
    s = time.time()
    # with open(MODEL_PATH + f'{model_name}_classifier_{cycle}.pkl', 'rb') as f:
    #     best_model = pickle.load(f)
    val_score = model.score(X_val, y_val)
    y_pred = model.predict(X_val)
    e = time.time()
    f = open(f'{signal_type}_{model_name}_report.txt','a') if save else save
    print(f'- Acurácia no conjunto de validação: {val_score * 100:.2f}%')
    print(f'- Tempo necessário para predição do conjunto de validação: {np.round(e - s, 3)} segundos')
    return y_pred, val_score * 100, np.round(e - s, 3)

def generate_confusion_matrix(y_val, y_pred, image_path, filename, title='', colorscale='blues',
                              width=500, height=500):
    data = {'Real':    y_val,
            'Predito': y_pred}
    df = pd.DataFrame(data, columns=['Real','Predito'])
    confusion_matrix = pd.crosstab(df['Real'], df['Predito'], rownames=['Real'],
                                   colnames=['Predito'], margins = True)
    cm = confusion_matrix.drop('All', axis=1).drop('All', axis=0)

    # Inverte rows because create_annotated_heatmap creates matrix in inverted order
    c = cm.values[::-1]
    x = list(cm.index)
    y = x[::-1]
    c_text = [[str(y) for y in x] for x in c]

    fig = ff.create_annotated_heatmap(c, x=x, y=y, annotation_text=c_text, colorscale=colorscale)

    # add title
    fig.update_layout(title_text=f'<i><b>Matriz de Confusão {title}</b></i>',
                      title_x=0.5, autosize=False, width=width, height=height,)

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14), x=0.5, y=-0.12, showarrow=False,
                            text="Valores Preditos", xref="paper", yref="paper"))

    fig.add_annotation(dict(font=dict(color="black",size=14), x=-0.2, y=0.5, textangle=270,
                            showarrow=False, text="Valores Reais", xref="paper", yref="paper"))
    fig.write_image(image_path + filename + '.svg')

def generate_title(cycle, model_name):
    title = cycle.split('_')[-1]
    if title != '1':
        title = f'{model_name.title()} e 1/{title} ciclo pós falta'
    else:
        title = f'{model_name.title()} e 1 ciclo pós falta'
    return title

def training_rf(signal_type):
    INPUT_DATA_PATH = '../input-data/'

    cycles = ['cycle_1', 'cycle_2', 'cycle_4', 'cycle_8', 'cycle_16', 'cycle_32', 'cycle_64',
              'cycle_128']

    model_name = 'random_forest'
    print(f'# Treinamento do modelo usando RandomForest dataset1 sinal: {signal_type}')
    for cycle in cycles:
        print('\n---')
        c = cycle.split('_')[-1]
        if c == '1':
            title = f'\n## {c} Ciclo Pós Falta'
        else:
            title = f'\n## 1/{c} Ciclo Pós Falta'
        print(title)
        train_X = open_folds(cycle, 'train', 'X', signal_type)
        train_y = open_folds(cycle, 'train', 'y', signal_type)
        test_X = open_folds(cycle, 'test', 'X', signal_type)
        test_y = open_folds(cycle, 'test', 'y', signal_type)

        X_train = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/X_train')
        y_train = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/y_train')
        X_val = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/X_val')
        y_val = decompress_pickle(INPUT_DATA_PATH + f'folds/{signal_type}/{cycle}/y_val')

        max_list = find_max_value(X_train)
        X_train_norm = normalizing(X_train, max_list)
        X_val_norm = normalizing(X_val, max_list)

        X_train_norm = transform_tabular_data(X_train_norm)
        X_val_norm = transform_tabular_data(X_val_norm)
        index = int(X_train_norm.shape[1] / 4)
        X_train_norm = apply_notch(notch_filter, X_train_norm, index)
        X_val_norm = apply_notch(notch_filter, X_val_norm, index)

        rf, random_grid = ref_grid()
        model1_best = param_optimization(X_train_norm, y_train, rf, random_grid)
        print(model1_best)
        model = RandomForestClassifier(**model1_best)
        list(map(lambda X: normalizing(X, max_list), train_X))
        list(map(lambda X: normalizing(X, max_list), test_X))

        train_X = transform_tabular_data(train_X, is_list=True)
        test_X = transform_tabular_data(test_X, is_list=True)

        new_train_X = []
        for data_X in train_X:
            samples_df = apply_notch(notch_filter, data_X, index)
            new_train_X.append(samples_df)

        new_test_X = []
        for data_X in test_X:
            samples_df = apply_notch(notch_filter, data_X, index)
            new_test_X.append(samples_df)

        scores = kfold(new_train_X, train_y, new_test_X, test_y, model, cycle, max_list, signal_type,
                    model_name=model_name, save=True)
        y_pred, val_acc, val_time = validating(X_val_norm, y_val, model, model_name, signal_type, save=True)
        title = generate_title(cycle, model_name)
        generate_confusion_matrix(y_val, y_pred, 'figs_cm_rf/', f'{signal_type}_{cycle}_{model_name}', title=title)
    print('*' * 100)
# if __name__ == '__main__':
    
