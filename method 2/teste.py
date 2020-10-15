model1_best = {'bootstrap': False,
 'criterion': 'entropy',
 'max_depth': 40,
 'max_features': 'auto',
 'min_samples_leaf': 2,
 'min_samples_split': 5,
 'n_estimators': 100}
with open("report.txt", "a") as file:
    file.write("================== RELATÓRIO 1 CICLO PÓS FALTA ==================\n")
    file.write(f"Model parameters:\n")
    for k, v in model1_best.items():
        file.write(f'\t{k}: {v}\n')
        # file.write(str(k) + ' >>> '+ str(v) + '\n')
    file.write("\nAcurácia média em cada um dos folds após repetição de 100 vezes:\n")
    for counter, value in enumerate(range(5, 11)):
        file.write(f'\tFold {counter + 1}: {value * 100:.2f}%\n')
    file.write(f'Média da acurácia: %\n')
    file.write(f'Desvio padrão da acurácia: %\n\n')