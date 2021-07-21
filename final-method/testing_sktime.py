from training_sktime import RidgeClassifierCV, MiniRocketMultivariate, Rocket, np, training

cycles = ['cycle_1', 'cycle_2', 'cycle_4', 'cycle_8', 'cycle_16', 'cycle_32']

print('# Treinamento do modelo usando MiniRocket')
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
        training(signal, cycle, model, model_name, transformation, save=True)
    print(f'\n### Treinando com 10000 features (default)', sep='')
    transformation = MiniRocketMultivariate(random_state=42)
    model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    signal, model_name = 'i', 'minirocket'
    training(signal, cycle, model, model_name, transformation, save=True)

print('*' * 100)

print('# Treinamento do modelo usando Rocket')
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
        training(signal, cycle, model, model_name, transformation, save=True)
    print(f'\n### Treinando com 10000 kernels (default)', sep='')
    transformation = Rocket(random_state=42)
    model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    signal, model_name = 'i', 'rocket'
    training(signal, cycle, model, model_name, transformation, save=True)

print('*' * 100)

print('# Treinamento do modelo usando Rocket e todos cores da CPU')
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
        transformation = Rocket(num_kernels=num_kernels, random_state=42, n_jobs=-1)
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        training(signal, cycle, model, model_name, transformation, save=True)
    print(f'\n### Treinando com 10000 kernels (default)', sep='')
    transformation = Rocket(random_state=42, n_jobs=-1)
    model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    signal, model_name = 'i', 'rocket'
    training(signal, cycle, model, model_name, transformation, save=True)