from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import sys
sys.path.append('../src/')
from functions import (data_list, join_signal, param_optimization, train_model,
                       gen_report, save_confusion_matrix)

data_path = '../data/detected_signals/'
model_path = 'models/'
fig_path = 'figs/'

data_list = data_list(data_path)
key = 'sinal_notch_final_current'
data_1_ciclo = join_signal(1, data_list, key)
data_1_2_ciclo = join_signal(2, data_list, key)
data_1_4_ciclo = join_signal(4, data_list, key)
data_1_8_ciclo = join_signal(8, data_list, key)
data_1_16_ciclo = join_signal(16, data_list, key)
data_1_32_ciclo = join_signal(32, data_list, key)

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
