from training_sktime import RidgeClassifierCV, MiniRocketMultivariate, Rocket, np, training

def training_sktime():
    cycles = ['cycle_1', 'cycle_2', 'cycle_4', 'cycle_8', 'cycle_16', 'cycle_32', 'cycle_64',
            'cycle_128']

    header = '|Ciclos pós falta| nº de features | Acurácia média de treinamento (%) | Acurácia de validação (%) | Tempo de treinamento (s) | Tempo de validação (s) |'
    sep = '\n|:---:|:---:|:---:|:---:|:---:|:---:|'
    with open("relatorio_automatizado_minirocket.md", 'w') as file:
        file.write(header)
        file.write(sep)

    print('# Treinamento do modelo usando MiniRocket dataset 1')
    for cycle in cycles:
        print('\n---')
        c = cycle.split('_')[-1]
        if c == '1':
            title = f'\n## {c} Ciclo Pós Falta'
        else:
            title = f'\n## 1/{c} Ciclo Pós Falta'
        print(title)
        signal, model_name = 'i', 'minirocket'
        for num_features in range(100, 1100, 100):
            print(f'\n### Treinando com {num_features} features', sep='')
            transformation = MiniRocketMultivariate(num_features=num_features, random_state=42)
            model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
            mean_acc, val_acc, train_time, val_time = training(signal, cycle, model, model_name, transformation, save=True)
            row = f'\n|{title.split(" ")[1]}|{num_features}|{mean_acc:.2f}|{val_acc:.2f}|{train_time}|{val_time}|'
            # Appending to file
            with open("relatorio_automatizado_minirocket.md", 'a') as file:
                file.write(row)
        print('\n### Treinando com 10000 features (default)', sep='')
        transformation = MiniRocketMultivariate(random_state=42)
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        signal, model_name = 'i', 'minirocket'
        mean_acc, val_acc, train_time, val_time = training(signal, cycle, model, model_name, transformation, save=True)
        row = f'\n|{title.split(" ")[1]}|10000|{mean_acc:.2f}|{val_acc:.2f}|{train_time}|{val_time}|'
        with open("relatorio_automatizado_minirocket.md", 'a') as file:
            file.write(row)

    print('*' * 100)

    header = '|Ciclos pós falta| nº de features | Acurácia média de treinamento (%) | Acurácia de validação (%) | Tempo de treinamento (s) | Tempo de validação (s) |'
    sep = '\n|:---:|:---:|:---:|:---:|:---:|:---:|'
    with open("relatorio_automatizado_rocket.md", 'w') as file:
        file.write(header)
        file.write(sep)

    print('# Treinamento do modelo usando Rocket dataset 1')
    for cycle in cycles:
        print('\n---')
        c = cycle.split('_')[-1]
        if c == '1':
            title = f'\n## {c} Ciclo Pós Falta'
        else:
            title = f'\n## 1/{c} Ciclo Pós Falta'
        print(title)
        signal, model_name = 'i', 'rocket'
        for num_kernels in range(100, 1100, 100):
            print(f'\n### Treinando com {num_kernels} kernels', sep='')
            transformation = Rocket(num_kernels=num_kernels, random_state=42)
            model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
            mean_acc, val_acc, train_time, val_time = training(signal, cycle, model, model_name, transformation, save=True)
            row = f'\n|{title.split(" ")[1]}|{num_kernels}|{mean_acc:.2f}|{val_acc:.2f}|{train_time}|{val_time}|'
            # Appending to file
            with open("relatorio_automatizado_rocket.md", 'a') as file:
                file.write(row)
        print('\n### Treinando com 10000 kernels (default)', sep='')
        transformation = Rocket(random_state=42)
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        signal, model_name = 'i', 'rocket'
        mean_acc, val_acc, train_time, val_time = training(signal, cycle, model, model_name, transformation, save=True)
        row = f'\n|{title.split(" ")[1]}|10000|{mean_acc:.2f}|{val_acc:.2f}|{train_time}|{val_time}|'
        # Appending to file
        with open("relatorio_automatizado_rocket.md", 'a') as file:
            file.write(row)

    print('*' * 100)