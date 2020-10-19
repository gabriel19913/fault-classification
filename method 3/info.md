# Explicação do método
---
Sinal usado:
- Sinal de corrente sem ruído.

Pré-processamento:
- Filtro notch

Variação do sinal:
- 1/4 de ciclo pré-falta e 1, 1/2, 1/4, 1/8, 1/16 e 1/32 ciclo pós-falta

## Validação dos modelos
### 1 CICLO PÓS FALTA 
*Model parameters:*
- n_estimators: 60
- min_samples_split: 5
- min_samples_leaf: 2
- max_features: sqrt
- max_depth: 30
- criterion: entropy
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 95.21%
- Fold 2: 91.94%
- Fold 3: 91.05%
- Fold 4: 91.14%
- Fold 5: 91.24%
- Fold 6: 90.80%
- Fold 7: 91.02%
- Fold 8: 91.42%
- Fold 9: 91.83%
- Fold 10: 91.91%

**Média da acurácia: 91.76%**

**Desvio padrão da acurácia: 1.21%**

**Tempo necessário para treinamento do modelo: 00:39:30**

### 1/2 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 20
- min_samples_split: 10
- min_samples_leaf: 1
- max_features: auto
- max_depth: 50
- criterion: entropy
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 92.89%
- Fold 2: 91.98%
- Fold 3: 91.84%
- Fold 4: 91.55%
- Fold 5: 91.06%
- Fold 6: 90.84%
- Fold 7: 91.05%
- Fold 8: 90.78%
- Fold 9: 91.15%
- Fold 10: 91.07%

**Média da acurácia: 91.42%**

**Desvio padrão da acurácia: 0.62%**

**Tempo necessário para treinamento do modelo: 00:12:07**


### 1/4 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 100
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: sqrt
- max_depth: 50
- criterion: entropy
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 91.44%
- Fold 2: 91.72%
- Fold 3: 90.31%
- Fold 4: 90.59%
- Fold 5: 91.58%
- Fold 6: 90.80%
- Fold 7: 90.76%
- Fold 8: 90.67%
- Fold 9: 90.48%
- Fold 10: 90.47%

**Média da acurácia: 90.88%**

**Desvio padrão da acurácia: 0.48%**

**Tempo necessário para treinamento do modelo: 00:41:21**

### 1/8 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 40
- min_samples_split: 5
- min_samples_leaf: 1
- max_features: auto
- max_depth: 40
- criterion: gini
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 85.39%
- Fold 2: 86.93%
- Fold 3: 86.89%
- Fold 4: 87.41%
- Fold 5: 88.09%
- Fold 6: 88.43%
- Fold 7: 88.72%
- Fold 8: 88.75%
- Fold 9: 88.43%
- Fold 10: 88.65%

**Média da acurácia: 87.77%**

**Desvio padrão da acurácia: 1.04%**

**Tempo necessário para treinamento do modelo: 00:07:00**

### 1/16 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 100
- min_samples_split: 5
- min_samples_leaf: 1
- max_features: sqrt
- max_depth: 50
- criterion: gini
- bootstrap: True

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 87.15%
- Fold 2: 88.10%
- Fold 3: 86.87%
- Fold 4: 86.56%
- Fold 5: 86.55%
- Fold 6: 86.58%
- Fold 7: 86.11%
- Fold 8: 86.33%
- Fold 9: 86.00%
- Fold 10: 85.99%

**Média da acurácia: 86.63%**

**Desvio padrão da acurácia: 0.61%**

**Tempo necessário para treinamento do modelo: 00:10:58**

### 1/32 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 50
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: auto
- max_depth: 20
- criterion: gini
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 77.46%
- Fold 2: 81.92%
- Fold 3: 82.28%
- Fold 4: 83.64%
- Fold 5: 84.92%
- Fold 6: 84.62%
- Fold 7: 84.78%
- Fold 8: 84.17%
- Fold 9: 84.11%
- Fold 10: 84.41%

**Média da acurácia: 83.23%**

**Desvio padrão da acurácia: 2.15%**

**Tempo necessário para treinamento do modelo: 00:08:08**