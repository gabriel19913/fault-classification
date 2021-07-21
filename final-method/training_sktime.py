import numpy as np
from sklearn.linear_model import RidgeClassifierCV
import pandas as pd
from sktime.transformations.panel.rocket import Rocket, MiniRocketMultivariate
import pickle
from glob import glob
import time
from noise import decompress_pickle
import plotly.figure_factory as ff

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

def evaluating_model(model, X_test, y_test, cycle, scores, count, model_name='model', save=None):
    # Evaluating model
    score = model.score(X_test, y_test)
    scores.append(score)
    if save:
        if len(scores) != 1 and score > scores[count] or len(scores) == 1:
            pickle.dump(model, open(MODEL_PATH + f'{model_name}_{cycle}.pkl', 'wb'))
    return scores

def print_results(cycle, model_name, scores, end_time, start_time, save=None):
    folds_labels = [f'- Fold {i}' for i in range(1, 11)]
    f = open(f'{model_name}_report.txt','a') if save else save
    title = generate_title(cycle, model_name)
    # print(f'\nAcurácia em cada fold usando {title}:', file=f)
    # for k, v in dict(zip(folds_labels, np.round(scores * 100, decimals=2))).items():
    #     print(f'{k:<7}: {v:^7.2f}%', file=f)
    # print(f'\nMédia da acurácia: {np.mean(scores) * 100:.2f}%', file=f)
    # print(f'Desvio padrão da acurácia: {np.std(scores) * 100:.2f}%', file=f)
    # print(f'Tempo necessário para treinamento: {np.round(end_time - start_time, 3)} segundos', file=f)
    print(f'\nAcurácia em cada fold:\n')
    for k, v in dict(zip(folds_labels, np.round(scores * 100, decimals=2))).items():
        print(f'{k:<7}: {v:^7.2f}%')
    print('\nO resulto final obtido foi:\n')
    print(f'- Média da acurácia: {np.mean(scores) * 100:.2f}%')
    print(f'- Desvio padrão da acurácia: {np.std(scores) * 100:.2f}%')
    print(f'- Tempo necessário para treinamento: {np.round(end_time - start_time, 3)} segundos')

def kfold(train_X, train_y, test_X, test_y, max_list, model, cycle, model_name='',
          transformation=None, save=None):
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
        scores = evaluating_model(model, X_te_transform, y_te, cycle, scores, count, model_name, save)

    e = time.time()
    final_scores = np.array(scores)
    print_results(cycle, model_name, final_scores, e, s, save)

def validating(X_val, y_val, model_name, cycle, save=None):
    s = time.time()
    with open(MODEL_PATH + f'{model_name}_{cycle}.pkl', 'rb') as f:
        best_model = pickle.load(f)
    val_score = best_model.score(X_val, y_val)
    y_pred = best_model.predict(X_val)
    e = time.time()
    f = open(f'{model_name}_report.txt','a') if save else save
    # print('*' * 73, file=f)
    print(f'- Acurácia no conjunto de validação: {val_score * 100:.2f}%')
    print(f'- Tempo necessário para predição do conjunto de validação: {np.round(e - s, 3)} segundos')
    # print('*' * 73, file=f)
    return y_pred

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

def training(signal, cycle, model, model_name='', transformation=None, save=None):
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
    kfold(train_X, train_y, test_X, test_y, max_list, model, cycle, model_name, transformation, save)
    y_pred = validating(X_val_transform, y_val, model_name, cycle, save)
    title = generate_title(cycle, model_name)
    generate_confusion_matrix(y_val, y_pred, 'figs_cm/', f'{cycle}_{model_name}', title=title)
    # print(f'Finalizado treinamento para {title}!')

# if __name__ == '__main__':
#     num_features = 2000
#     transformation = MiniRocketMultivariate(num_features=num_features)
#     model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
#     signal, model_name = 'i', 'minirocket'
#     cycle = 'cycle_1' # 'cycle_1', 'cycle_2', 'cycle_4', 'cycle_8', 'cycle_16', 'cycle_32'
#     training(signal, cycle, model, model_name, transformation, save=True)