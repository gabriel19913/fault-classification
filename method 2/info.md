# Explicação do método
---
Sinal usado:
- Sinal de tensão sem ruído.

Pré-processamento:
- Filtro notch

Variação do sinal:
- 1/4 de ciclo pré-falta e 1, 1/2, 1/4, 1/8, 1/16 e 1/32 ciclo pós-falta

## Validação dos modelos
### 1 CICLO PÓS FALTA 
*Model parameters:*
- n_estimators: 100
- min_samples_split: 5
- min_samples_leaf: 1
- max_features: auto
- max_depth: 50
- criterion: entropy
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 93.28%
- Fold 2: 93.26%
- Fold 3: 92.33%
- Fold 4: 94.14%
- Fold 5: 94.14%
- Fold 6: 93.82%
- Fold 7: 93.31%
- Fold 8: 93.50%
- Fold 9: 94.04%
- Fold 10: 94.11%

**Média da acurácia: 93.59%**

**Desvio padrão da acurácia: 0.55%**

**Tempo necessário para treinamento do modelo: 01:04:10**

### 1/2 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 100
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: sqrt
- max_depth: 20
- criterion: gini
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 94.73%
- Fold 2: 95.05%
- Fold 3: 94.03%
- Fold 4: 95.02%
- Fold 5: 94.55%
- Fold 6: 94.09%
- Fold 7: 94.42%
- Fold 8: 94.07%
- Fold 9: 93.91%
- Fold 10: 93.91%

**Média da acurácia: 94.38%**

**Desvio padrão da acurácia: 0.42%**

**Tempo necessário para treinamento do modelo: 00:26:50**


### 1/4 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 30
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: sqrt
- max_depth: 30
- criterion: entropy
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 86.82%
- Fold 2: 89.96%
- Fold 3: 90.87%
- Fold 4: 91.55%
- Fold 5: 91.50%
- Fold 6: 91.36%
- Fold 7: 92.05%
- Fold 8: 92.28%
- Fold 9: 92.49%
- Fold 10: 92.38%

**Média da acurácia: 91.13%**

**Desvio padrão da acurácia: 1.61%**

**Tempo necessário para treinamento do modelo: 00:14:20**

### 1/8 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 100
- min_samples_split: 5
- min_samples_leaf: 1
- max_features: sqrt
- max_depth: 50
- criterion: gini
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 93.35%
- Fold 2: 93.15%
- Fold 3: 92.55%
- Fold 4: 93.06%
- Fold 5: 93.09%
- Fold 6: 93.24%
- Fold 7: 93.16%
- Fold 8: 92.57%
- Fold 9: 92.78%
- Fold 10: 92.94%

**Média da acurácia: 92.99%**

**Desvio padrão da acurácia: 0.26%**

**Tempo necessário para treinamento do modelo: 00:20:06**

### 1/16 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 90
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: auto
- max_depth: 50
- criterion: entropy
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 91.78%
- Fold 2: 90.47%
- Fold 3: 92.75%
- Fold 4: 91.70%
- Fold 5: 92.22%
- Fold 6: 91.55%
- Fold 7: 91.65%
- Fold 8: 91.50%
- Fold 9: 91.34%
- Fold 10: 91.41%

**Média da acurácia: 91.64%**

**Desvio padrão da acurácia: 0.56%**

**Tempo necessário para treinamento do modelo: 00:36:30**

### 1/32 CICLO PÓS FALTA
*Model parameters:*
- n_estimators: 50
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: auto
- max_depth: 50
- criterion: gini
- bootstrap: False

*Acurácia média em cada um dos folds após repetição de 100 vezes:*
- Fold 1: 93.48%
- Fold 2: 92.23%
- Fold 3: 91.35%
- Fold 4: 91.81%
- Fold 5: 91.83%
- Fold 6: 90.75%
- Fold 7: 89.89%
- Fold 8: 89.88%
- Fold 9: 89.82%
- Fold 10: 90.02%

**Média da acurácia: 91.11%**

**Desvio padrão da acurácia: 1.18%**

**Tempo necessário para treinamento do modelo: 00:09:26**