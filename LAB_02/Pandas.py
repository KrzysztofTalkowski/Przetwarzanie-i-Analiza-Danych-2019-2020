#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[7]:


dates = pd.date_range('20200301', periods = 5)
dates


# In[17]:


df = pd.DataFrame(np.random.rand(5,3), index =dates, columns=list('ABC'))
df.index.name = 'data'
df


# In[60]:


# liczby losowe integer w zakresie od 1 dla 100 (dla porządku)
df =  pd.DataFrame(np.random.randint(0, 100,(20,3)), columns=list('ABC'))
df.head(3)


# In[61]:


df.tail(3)


# In[62]:


indexNamesArr = df.index.values
list(indexNamesArr)


# In[63]:


columnsNamesArr = df.columns.values
list(columnsNamesArr)


# In[64]:


df.values


# In[65]:


df.sample(5)


# In[66]:


df.loc[0:19,"A"]


# In[68]:


df.loc[:,:"B"]


# In[72]:


df.iloc[:3, :2]


# In[75]:


df.iloc[4]


# In[101]:


df.iloc[[0,5,6,7],0:2]


# In[103]:


df.describe()


# In[104]:


df.loc[:,(df>0).all()] 


# In[120]:


df.mean


# In[123]:


df.mean(axis=1)


# In[131]:


df1 = pd.DataFrame(np.random.rand(5,3), index =dates, columns=list('ABC'))
df1


# In[135]:


df2 = pd.concat([df1,df])
df2.transpose()


# In[143]:


d = {'x': [1, 2, 3, 4, 5], 'y': ["a", "b", "a", "b", "b"]}

df = pd.DataFrame(data=d,index=np.arange(5))
df


# In[153]:


df.index.name = 'id'
df.sort_values(by='id',ascending= True)


# In[154]:


df.sort_values(by='y',ascending=False)


# In[160]:


slownik = {'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'], 'Fruit': ['Apple',
'Apple', 'Banana', 'Banana', 'Apple'], 'Pound': [10, 15, 50, 40, 5], 'Profit':[
20, 30, 25, 20, 10]}
df3 = pd.DataFrame(slownik)
df3


# In[163]:


df3.groupby('Day').sum()


# In[166]:


df3.groupby(['Day','Fruit']).sum()


# In[168]:


df3.groupby(['Fruit','Day']).sum()


# In[171]:


df=pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
df.index.name='id'
df


# In[174]:


#Zmiana wszystkich wartości w kolumnie B na 1
df['B'] = 1
df


# In[177]:


# zmiana konkretnej wartości w tabeli na 10, w tym wypadku 2gi wiersz, 3cia kolumna [1,2]
df.iloc[1,2] = 10
df


# In[180]:


# Tutaj nie mam pojęcia - ani średnik, ani "¡" nie działają
# Prawdopodobnie chcielibyśmy przypisać wartości przeciwne do jakiejś części tabeli, ale nie będę strzelał
df[df¡0] = -df
df


# In[182]:


# zmiana wartości w pierwszym i czwartym wierszu drugiej kolumny na NaN
# NaN = Not a Number = any value that is undefined or unpresentable
df.iloc[[0,3],1] = np.nan
df


# In[184]:


# zamiana wszystkich występowań NaN na 0
df.fillna(0, inplace= True)
df


# In[188]:


# zamiana wszystkich występowań NaN na -9999
df.iloc[[0, 3], 1] = np.nan
df=df.replace(to_replace=np.nan,value=-9999)
df


# In[191]:


# Sprawdź które jest null czyli w którym miejsce jest 'puste'
df.iloc[[0, 3], 1] = np.nan
print(pd.isnull(df))


# In[195]:


d = {'x': [1, 2, 3, 4, 5], 'y': ["a", "b", "a", "b", "b"]}
df = pd.DataFrame(data=d)
df


# In[208]:


# w zmiennej Y nie ma wartości liczbowych, więc nie można policzyć średniej
print(df.groupby(['y']).mean())


# In[218]:


df['y'].value_counts()


# In[228]:


df['x'].value_counts()


# In[243]:


import os
import csv
#Przy wprowadzeniu danych z pliku za pomocą np.load musimy 
#podać wszystkie rubryki tabeli ,przy wczytywaniu za pomocą pd.read_csv('autos.csv')
#nie trzeba - są automatycznie pogrupowane
autos = pd.read_csv(r"C:\Users\Krzysiek\Desktop\autos.csv")
autos


# In[247]:


fuel=autos.groupby('make').mean()[['city-mpg','highway-mpg']]
fuel


# In[249]:


print(autos.groupby('make')['fuel-type'].value_counts())


# In[254]:


a1 = np.polyfit(autos['length'], autos['city-mpg'], 1)
a2 = np.polyfit(autos['length'], autos['city-mpg'], 2)
a1
a2


# In[257]:


from scipy import stats as st
st.pearsonr(autos['length'], autos['city-mpg'])


# In[261]:


import matplotlib.pyplot as plt
xes = np.linspace(autos['length'].min(), autos['length'].max(), 500)
plt.scatter(autos['length'], autos['city-mpg'], label = 'próbki')
plt.xlabel('Car length')
plt.ylabel('City mpg')
plt.plot(xes, np.polyval(a1, xes), label = 'x')
plt.plot(xes, np.polyval(a2, xes), label = 'x^2')
plt.legend()


# In[264]:


gauss = st.gaussian_kde(autos['length'])
plt.figure()
plt.plot(xes, gauss(xes), label = 'estymator')
plt.scatter(autos['length'], gauss(autos['length']), label = 'próbki')
plt.legend()


# In[268]:


plt.figure()
ax = plt.subplot(2, 1, 1)
ax.plot(xes, gauss(xes), label = 'Estymator')
ax.scatter(autos['length'], gauss(autos['length']), label = 'próbki')
ax.set_title("Length")
plt.legend()

xesWidth = np.linspace(autos['width'].min(), autos['width'].max(), 500)
gaussWidth = st.gaussian_kde(autos['width'])

ax2 = plt.subplot(2, 1, 2)
ax2.plot(xesWidth, gaussWidth(xesWidth), label = 'Estymator')
ax2.scatter(autos['width'], gaussWidth(autos['width']), label = 'próbki')
ax2.set_title("Width")
plt.legend()


# In[289]:


xmin, xmax, ymin, ymax = autos['width'].min(), autos['width'].max(), autos['length'].min(), autos['length'].max()
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([autos['width'], autos['length']])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure()
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='Greens')
cset = ax.contour(xx, yy, f, colors='b')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Length')
ax.set_ylabel('Width')

plt.savefig('wynik.png')
plt.savefig('wynik.pdf')

plt.show()

