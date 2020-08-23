# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:20:01 2020

@authors: Aldo and Giorgio
"""

import pandas as pd

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statistics
import seaborn as sns
from collections import Counter 




df = pd.read_csv('./FinalTest.csv')
df.fillna(method='ffill',inplace=True)
df.fillna(method='bfill',inplace=True)

#Giorno infezione a partire da day 0 fino al 23esimo, principali zone cina. 1/22/2020 data effettiva
print("\n\n")



# Estraggo province
Province = df['Province/State'].values.tolist()
print("Principali province interessate \n")
print (*Province,sep = "\n")
val = input("Inserisci una provincia dalla lista elencata per maggiori informazioni -> -> ->  ")
val = val.title()



index = Province.index(val)
#print("Index city",index)
Infected=(df.iloc[index,:]).tolist()
Infected.pop(0)


l= list(range(24))


corr, _ = pearsonr(Infected,l)
print('Indice Pearson: %.3f' % corr)


#Costruisco regressore Lineare
l=np.array(l,dtype = int)
Infected=np.array(Infected,dtype = int)


X = np.reshape(l, (-1, 1))
Y = np.reshape(Infected, (-1, 1))
regsr=LinearRegression()
regsr.fit(X,Y)

#input controllato
while True:
    try:
        DayX = int(input('Inserisci un giorno maggiore di 24 per la previsione -> -> -> :  '))
        if DayX > 23:
            break
        else:
            print ('Numero errato')
    except ValueError:
        print ('Input was not a digit - please try again.')

       
to_predict_x= [DayX]
to_predict_x= np.array(to_predict_x).reshape(-1,1)
predicted_y= regsr.predict(to_predict_x)
m= regsr.coef_
c= regsr.intercept_

print("Infetti Previsti a {} : {}\n".format(val,int(predicted_y)))
print("Inclinazione (m): ",m)
print("Intercetta (c): ",c)


plt.title('Predizione Numero Infetti')  
plt.xlabel('Days')  
plt.ylabel('Infecteds') 
plt.scatter(X,Y,color="blue")

new_y=[ m*i+c for i in np.append(X,to_predict_x)]
new_y=np.array(new_y).reshape(-1,1)
plt.plot(np.append(X,to_predict_x),new_y,color="red")
plt.show()
#FINE PREDIZIONE


input("Premi ENTER per visualizzare il plot e altre informazioni")

#Plot NORMALE DEI DATI
days=range(0,24)
plt.plot(days,Infected)

plt.xlabel('days')
plt.ylabel('Infected')
plt.show()
#Altre statistiche

#MEDIA
mean = Infected.mean()
print("Media : {} ".format(mean))

#MEDIANA
n = len(Infected) 
Infected.sort() 
if n % 2 == 0: 
    median1 = Infected[n//2] 
    median2 = Infected[n//2 - 1] 
    median = (median1 + median2)/2
else: 
    median = Infected[n//2] 
print("Median is: {} ".format(median))

#MODA
d_mem_count = Counter(Infected)
for k in d_mem_count.keys():
 if d_mem_count[k] > 1: 
  print("Mode is : {} ".format(k))


#INFORMAZIONI GENERALI SULLE PROVINCIE
print("Media infetti totale di tutte le province  :{} \n".format(int(df["23"].mean())))  # ULTIMO GIORNO ANALIZZATO

print("Last tracking")
newDf = df[['Province/State','23']].copy()
print(newDf)

'''
bycity= newDf.groupby(by="Province/State")
avg_infects= bycity.mean();
print(avg_infects);
'''