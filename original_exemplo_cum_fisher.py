# -*- coding: utf-8 -*-
"""original exemplo_cum_fisher.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JKpZXWsaXFnm21hpxkbDJHHbXWPjfEZe

# EXEMPLO DO USO DOS CUMULANTES 
 Este exemplo usa os cumulantes de ordem 2, 3 e 4 para extrair características de sinais elétricos,
 após a extração, as melhores características são selecionadas por meio da razão discriminante de fisher.


---


## PASSO 1:

Geração dos dados

Descrição: 
- Classe 1: sinal sem disturbio
- Classe 2: sinal com disturbio (swell neste caso)

Importando as bibliotecas necessárias
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

"""Função para gerar sinais com distúrbios:"""

def swellEvent(fundFreq, nPointsCycle, nCycles, SNRdb, nEvents):

	# Constants
	PI = np.pi
	fundAmp = 1
	phaseA = np.random.uniform(-PI, PI, nEvents)
	phaseB = phaseA + 2*PI/3
	phaseC = phaseA + 4*PI/3
	nPoints = nCycles*nPointsCycle
	samplingFreq = nPointsCycle*fundFreq
	samplingPeriod = 1/samplingFreq
	w0 = 2*PI*fundFreq
	distStartingPoint = np.random.uniform(0.3*nPoints, 0.7*nPoints, nEvents).astype(int)  #The disturbance starts in somewhere between 30-70% of the samples
	distPercentual = np.random.uniform(0.2, 0.9, nEvents) # The increase in voltage amp will be something between 20 and 90 per cent.
	distDuration = np.random.uniform(100, 200, nEvents).astype(int)
	distStep = distPercentual/(nPointsCycle) # Ain't it nPointsCycle/4, as in the interruption event?


	t = np.linspace(0, samplingPeriod*nPoints-samplingPeriod, nPoints)

	# Pre-allocating some space
	xA = np.zeros((nEvents,nPoints))
	xB = np.zeros((nEvents,nPoints))
	xC = np.zeros((nEvents,nPoints))
	xAdist = np.zeros((nEvents,nPoints))	
	xBdist = np.zeros((nEvents,nPoints))	
	xCdist = np.zeros((nEvents,nPoints))	

	# Noise generation:
	signalPower = (fundAmp**2)/2
	noisePower = signalPower/(10**(SNRdb/10))
	deviation = noisePower**(0.5)

	for i in range(nEvents):
		noiseA = np.random.normal(0, deviation, (1, nPoints)) # A média já é aproximadamente zero (<0.001), devo retirar a média dos dados novamente? (Como estava no MATLAB)
		noiseB = np.random.normal(0, deviation, (1, nPoints))
		noiseC = np.random.normal(0, deviation, (1, nPoints))

	# Three-Phase power signal generation:

	for i in range(nEvents):
		xA[i,:] = fundAmp*np.sin(w0*t + phaseA[i])
		xB[i,:] = fundAmp*np.sin(w0*t + phaseB[i])
		xC[i,:] = fundAmp*np.sin(w0*t + phaseC[i])

	# Disturbance generation

	for i in range(nEvents):
		fundAmp = 1
		for k in range(nPoints):
			if (k>distStartingPoint[i] and  k<(distStartingPoint[i]+nPointsCycle+1)):
				fundAmp = fundAmp + distStep[i]
			if ((k>(distStartingPoint[i]+distDuration[i])) and  (k<(distStartingPoint[i]+distDuration[i]+nPointsCycle+1))):
				fundAmp = fundAmp - distStep[i]

			xAdist[i,k] = fundAmp*np.sin(w0*t[k] + phaseA[i])
			xBdist[i,k] = fundAmp*np.sin(w0*t[k] + phaseB[i])
			xCdist[i,k] = fundAmp*np.sin(w0*t[k] + phaseC[i])

	xA = xA + noiseA
	xB = xB + noiseB
	xC = xC + noiseC
	xAdist = xAdist + noiseA
	xBdist = xBdist + noiseB
	xCdist = xCdist + noiseC

	return xA, xB, xC, xAdist, xBdist, xCdist

"""Gerando sinais com e sem distúrbios:"""

fundFreq = 60
nPointsCycle = 256
nCycles = 2
SNRdb = 30    # Ruído
nEvents = 50

xA, xB, xC, xAdist, xBdist, xCdist = swellEvent(fundFreq, nPointsCycle, nCycles, SNRdb, nEvents)

xA.shape

xA.shape

"""Visualização dos sinais:"""

plt.figure()
plt.title('Sinal trifásico sem distúrbio')
plt.plot(xA[0,:])
plt.plot(xB[0,:])
plt.plot(xC[0,:])
plt.xlabel("Amostra")
plt.ylabel("Amplitude(p.u)")
plt.show()

plt.figure()
plt.title('Sinal trifásico com distúrbio')
plt.plot(xAdist[0,:])
plt.plot(xBdist[0,:])
plt.plot(xCdist[0,:])
plt.xlabel("Amostra")
plt.ylabel("Amplitude(p.u)")
plt.show()

"""## PASSO 2: Cálculo dos cumulantes dos sinais

Funções para cálculo dos cumulantes:
"""

# OBS: Somente as três últimas funções são utilizadas, as primeiras são apenas
# funções auxiliares.

def cum2Calc(vetMediaZero, nPoints, ii):
	return np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii))/nPoints

def cum3Calc(vetMediaZero, nPoints, ii):
	return np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii)**2)/nPoints

def cum4Calc(vetMediaZero, nPoints, ii):
	sumOfSquares = np.dot(vetMediaZero.T, vetMediaZero)
	part1 = np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii)**3)/nPoints
	part2 = 3*np.matmul(vetMediaZero.T, np.roll(vetMediaZero, ii))*sumOfSquares/(nPoints**2)
	cum4 = part1 - part2
	return cum4

def cumCalc(vetEntrada, nEvents, order):

	if (order == 2): 
		functionCalled = cum2Calc
	elif (order == 3):
		functionCalled = cum3Calc
	elif (order==4):
		functionCalled = cum4Calc
	else:
		return

	# Transformando vetEntrada em um vetor coluna:
	dimVet = vetEntrada.shape
	if(dimVet[0]==nEvents):   
		vetEntrada = vetEntrada.T

	nPoints = vetEntrada[:,0].size  # number of points per column
	
	# Pre - allocating space
	cum = np.zeros([nEvents, nPoints])

	for i in range(nEvents):
		# Transformando vetEntrada em um vetor de média nula:
		media = np.mean(vetEntrada[:,i])
		vetMediaZero = vetEntrada[:,i] - media

		for ii in range(nPoints):
			cum[i,ii] = functionCalled(vetMediaZero, nPoints, ii)

	return cum

def cum2(vetEntrada, nEvents):
	return cumCalc(vetEntrada, nEvents, 2)

def cum3(vetEntrada, nEvents):
	return cumCalc(vetEntrada, nEvents, 3)

def cum4(vetEntrada, nEvents):
	return cumCalc(vetEntrada, nEvents, 4)

"""Extração dos cumulantes dos sinais com e sem distúrbio:"""

# Exemplo de cálculo dos cumulantes:
# função cum2(xA, nEvents) 
#	-> xA é o vetor de sinais cujos cumulantes serão calculados
#	-> nEvents é o número de eventos de sinais. 

nEvents = 50  # definido na simulação dos eventos

timeStart = time.perf_counter()
fundCum2 = cum2(xA, nEvents);  # cumulante de ordem 2 dos sinais fundamentais
fundCum3 = cum3(xA, nEvents);
fundCum4 = cum4(xA, nEvents);
timeElapsed = time.perf_counter() - timeStart
print("Cumulantes da classe 1 calculados!")
print("Time Elapsed: " + str(timeElapsed))

# Descartando metade dos cumulantes de ordem 2 devido à simetria
fundCum2 = fundCum2[:, int(fundCum2.shape[1]/2)] 

# Agrupando os cumulantes da classe 1 em um vetor
C1 = np.column_stack((fundCum2, fundCum3))     
C1 = np.column_stack((C1, fundCum4))

timeStart = time.perf_counter()
distCum2 = cum2(xAdist, nEvents); # cumulante de ordem 2 do sinal com disturbio
distCum3 = cum3(xAdist, nEvents);
distCum4 = cum4(xAdist, nEvents);
timeElapsed = time.perf_counter() - timeStart
print("Cumulantes da classe 2 calculados!")
print("Time Elapsed: " + str(timeElapsed))

# Descartando metade dos cumulantes de ordem 2 devido à simetria
distCum2 = distCum2[:, int(distCum2.shape[1]/2)] 

C2 = np.column_stack((distCum2, distCum3)) # dados da classe 2 (com distúrbio)
C2 = np.column_stack((C2, distCum4))

distCum2.shape

C2.shape

fundCum3.shape

"""## PASSO 3: Seleção de características

Função da razão discriminante de fisher:
"""

def fisherRatio(C, integer_encoded, nClasses):
    # Implementação do Fisher Multiclass, conforme apresentado em Theodoridis,
    # que dá a importância média de uma característica para problema de
    # classifição com múltiplas classes. Se reduz na formulação usual quando 
    # aplicado a um problema de classificação binária.
	classes = []
	for i in range(nClasses):
		classes.append(C[integer_encoded == i])
	
	classes = np.array(classes)
	
	m = np.zeros((nClasses, C.shape[1]))
	var = np.zeros((nClasses,  C.shape[1]))
	for i in range(nClasses):
		m[i] = np.mean(classes[i], axis=0)
		var[i] = np.var(classes[i], axis=0)
		
	J = np.zeros((nClasses, C.shape[1]))
	for i in range(nClasses):
		for j in range(nClasses):
			if(i != j):
				J[i] += np.divide(np.power(m[i] - m[j], 2), var[i] + var[j])
	
	J = np.sum(J, axis=0)			
	I = np.argsort(J, axis=0)
	I = I[::-1]
	return I, J

"""Selecionando algumas características com base na Razão Discriminante de Fisher:"""

# Criando um vetor de alvos
#	-> 0 = classe 1
#	-> 1 = classe 2
targets = np.zeros((nEvents, 1))
targets = np.row_stack((targets, np.ones((nEvents, 1))))
targets = targets[:,0]
C = np.row_stack((C1, C2))
I, J = fisherRatio(C, targets, nClasses)

# Selecionando 3 características:
indexes = [I[0], I[500], I [800]]


# Plotando o espaço de classificação com as três melhores características:
target_names = ['Classe sem distúrbio', 'Classe com distúrbio']
colors = ['navy', 'turquoise', 'darkorange', 'brown', 'red', 'gold', 'purple', 'blue', 'lime', 'black']
markers = ['o', 'p', 'x', '*', 's', 'D', '^', '1', 'v', 'd'  ]	
lw = 2


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for color, i, target_name, marker in zip(colors, range(nClasses), target_names, markers):
    ax.scatter(C[targets == i, indexes[0]], C[targets == i, indexes[1]], C[targets == i, indexes[2]], color=color, alpha=.8, lw=lw, label=target_name, marker = marker)
ax.set_xlabel("C1")
ax.set_ylabel("C2")
ax.set_zlabel("C3")
plt.show()